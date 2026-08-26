import asyncio

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_context_config
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger("index_data_points")


def _embeddable_value(data_point, field_name: str):
    """The exact value that would be embedded for ``field_name``."""
    return getattr(data_point, field_name, None)


async def _drop_unchanged(vector_engine, type_name: str, field_name: str, points: list):
    """Return only the points whose indexed field differs from what is stored.

    Re-embedding and upserting a point whose indexed text has not changed costs
    an embedding call and a write, and produces a byte-identical row. On stores
    that version on write (LanceDB upserts via ``merge_insert``) each redundant
    write also creates a new table version and data fragment, which accumulate
    until something compacts them.

    Fails OPEN in the direction of doing the work: any error, any point the
    store does not know, and any payload that does not carry the field is
    indexed as before. Skipping is only ever chosen on positive evidence that
    the stored value is identical, so this can never leave a vector stale.
    """
    ids = [str(point.id) for point in points if getattr(point, "id", None) is not None]
    if not ids:
        return points
    try:
        stored_rows = await vector_engine.retrieve(type_name, ids)

        stored_by_id: dict[str, dict] = {}
        for row in stored_rows or []:
            row_id = getattr(row, "id", None)
            payload = getattr(row, "payload", None)
            if row_id is None and isinstance(row, dict):
                row_id, payload = row.get("id"), row.get("payload")
            if row_id is not None and isinstance(payload, dict):
                stored_by_id[str(row_id)] = payload

        changed = []
        for point in points:
            payload = stored_by_id.get(str(getattr(point, "id", "")))
            if payload is None or field_name not in payload:
                changed.append(point)
                continue
            if payload[field_name] != _embeddable_value(point, field_name):
                changed.append(point)
        return changed
    except Exception:  # noqa: BLE001
        # ANY failure means index as before. A store that cannot answer, an
        # adapter without retrieve, a mocked engine returning something
        # unexpected: all of them must index, never skip.
        return points


async def index_data_points(data_points: list[DataPoint], vector_engine=None):
    """Index data points in the vector engine by creating embeddings for specified fields.

    Data points are indexed in batches of the embedding engine's batch_size. A semaphore
    bounds how many batches run concurrently so that at most
    ``embedding_max_concurrent_data_points`` data points (default 150, env
    ``EMBEDDING_MAX_CONCURRENT_DATA_POINTS``) are in flight at once:
    ``max(1, embedding_max_concurrent_data_points // batch_size)`` concurrent batches.

    Args:
        data_points: List of DataPoint objects to index. Each DataPoint's metadata must
                     contain an 'index_fields' list specifying which fields to embed.
        vector_engine: Optional pre-created vector engine. Falls back to
                       ``get_vector_engine_async()`` when not supplied.

    Returns:
        The original data_points list.
    """
    data_points_by_type = {}

    vector_engine = vector_engine or await get_vector_engine_async()

    for data_point in data_points:
        # Skip non-DataPoint objects (e.g. CogneeGraph) that may be
        # passed through the memify pipeline without metadata.
        if not hasattr(data_point, "metadata") or not data_point.metadata:
            continue

        data_point_type = type(data_point)
        type_name = data_point_type.__name__

        for field_name in data_point.metadata["index_fields"]:
            if getattr(data_point, field_name, None) is None:
                continue

            if type_name not in data_points_by_type:
                data_points_by_type[type_name] = {}

            if field_name not in data_points_by_type[type_name]:
                await vector_engine.create_vector_index(type_name, field_name)
                data_points_by_type[type_name][field_name] = []

            indexed_data_point = data_point.model_copy(deep=True)
            indexed_data_point.metadata["index_fields"] = [field_name]
            data_points_by_type[type_name][field_name].append(indexed_data_point)

    batch_size = vector_engine.embedding_engine.get_batch_size()
    max_concurrent_data_points = get_embedding_context_config().embedding_max_concurrent_data_points
    semaphore = asyncio.Semaphore(max(1, max_concurrent_data_points // batch_size))

    async def _index_batch(type_name, field_name, batch):
        async with semaphore:
            await vector_engine.index_data_points(type_name, field_name, batch)

    skip_unchanged = getattr(
        get_embedding_context_config(), "skip_unchanged_vector_writes", True
    )

    tasks = []
    for type_name, fields in data_points_by_type.items():
        for field_name, points in fields.items():
            if skip_unchanged:
                before = len(points)
                points = await _drop_unchanged(vector_engine, type_name, field_name, points)
                if len(points) != before:
                    logger.debug(
                        "index_data_points: %s.%s skipped %d unchanged of %d",
                        type_name,
                        field_name,
                        before - len(points),
                        before,
                    )
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                tasks.append(asyncio.create_task(_index_batch(type_name, field_name, batch)))

    await asyncio.gather(*tasks)

    return data_points


async def get_data_points_from_model(
    data_point: DataPoint, added_data_points=None, visited_properties=None
) -> list[DataPoint]:
    data_points = []
    added_data_points = added_data_points or {}
    visited_properties = visited_properties or {}

    for field_name, field_value in data_point:
        if isinstance(field_value, DataPoint):
            property_key = f"{str(data_point.id)}{field_name}{str(field_value.id)}"

            if property_key in visited_properties:
                return []

            visited_properties[property_key] = True

            new_data_points = await get_data_points_from_model(
                field_value, added_data_points, visited_properties
            )

            for new_point in new_data_points:
                if str(new_point.id) not in added_data_points:
                    added_data_points[str(new_point.id)] = True
                    data_points.append(new_point)

        if (
            isinstance(field_value, list)
            and len(field_value) > 0
            and isinstance(field_value[0], DataPoint)
        ):
            for field_value_item in field_value:
                property_key = f"{str(data_point.id)}{field_name}{str(field_value_item.id)}"

                if property_key in visited_properties:
                    return []

                visited_properties[property_key] = True

                new_data_points = await get_data_points_from_model(
                    field_value_item, added_data_points, visited_properties
                )

                for new_point in new_data_points:
                    if str(new_point.id) not in added_data_points:
                        added_data_points[str(new_point.id)] = True
                        data_points.append(new_point)

    if str(data_point.id) not in added_data_points:
        data_points.append(data_point)

    return data_points


if __name__ == "__main__":

    class Car(DataPoint):
        model: str
        color: str
        metadata: dict = {"index_fields": ["name"]}

    class Person(DataPoint):
        name: str
        age: int
        owns_car: list[Car]
        metadata: dict = {"index_fields": ["name"]}

    car1 = Car(model="Tesla Model S", color="Blue")
    car2 = Car(model="Toyota Camry", color="Red")
    person = Person(name="John", age=30, owns_car=[car1, car2])

    data_points = get_data_points_from_model(person)

    print(data_points)
