import asyncio
from uuid import UUID

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_context_config
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger("index_data_points")


def write_batch_size() -> int:
    """Rows per vector write: ``VECTOR_WRITE_BATCH_SIZE``, 128 by default.

    Independent of the embedding request size. A write batch larger than one request
    is embedded in consecutive requests by ``EmbeddingGateway``, so the store receives
    fewer, larger writes (fewer statements and commits) while every request stays
    within the provider's input limit.
    """
    return get_vectordb_config().vector_write_batch_size


def _embedding_request_size(vector_engine) -> int | None:
    """Texts the engine sends per embedding request, or None when it does not say."""
    embedding_engine = getattr(vector_engine, "embedding_engine", None)
    get_batch_size = getattr(embedding_engine, "get_batch_size", None)
    if get_batch_size is None:
        return None
    size = get_batch_size()
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return None
    return size


def indexing_semaphore(vector_engine) -> asyncio.Semaphore:
    """The bound on concurrent vector writes, shared by every index call of one storage step.

    ``EMBEDDING_MAX_CONCURRENT_DATA_POINTS`` caps the data points in flight to the
    embedding engine; divided by the texts one request carries, that is how many writes
    may run at once (150 / 32 = 4 by default). The requests of one write batch run one
    after another inside the gateway, so the cap holds whatever the write batch size is.
    ``add_data_points`` hands one semaphore to its node and edge indexing so the two
    together stay within the cap.
    """
    max_concurrent = get_embedding_context_config().embedding_max_concurrent_data_points
    request_size = _embedding_request_size(vector_engine) or write_batch_size()
    return asyncio.Semaphore(max(1, max_concurrent // request_size))


def _for_field(data_point: DataPoint, field_name: str) -> DataPoint:
    """The data point as the store should see it for ``field_name``.

    Adapters embed ``DataPoint.get_embeddable_data``, the FIRST index field, so a point
    with a single index field is handed over as it is. A point indexed on several fields
    gets one shallow copy per field with its own metadata dict; nothing here mutates the
    original. The deep copy this replaces rebuilt every chunk's entities and document for
    each write, and ``add_data_points`` made a second one before that.
    """
    index_fields = list(data_point.metadata.get("index_fields") or [])
    if index_fields == [field_name]:
        return data_point
    return data_point.model_copy(
        update={"metadata": {**data_point.metadata, "index_fields": [field_name]}}
    )


async def index_data_points(
    data_points: list[DataPoint],
    vector_engine=None,
    semaphore: asyncio.Semaphore | None = None,
):
    """Index data points in the vector engine by creating embeddings for specified fields.

    Each ``(type, index field)`` pair is one collection. Duplicate IDs keep their
    last input and merge their tags before embedding. Its points are written in
    batches of ``write_batch_size()`` rows; the engine embeds a batch (in requests of its
    own size) and upserts it. ``semaphore`` bounds how many batches are in flight at
    once; by default ``indexing_semaphore(vector_engine)``, which keeps at most
    ``embedding_max_concurrent_data_points`` (default 150, env
    ``EMBEDDING_MAX_CONCURRENT_DATA_POINTS``) data points at the embedding engine. A
    caller indexing several lists in one step passes one semaphore to all of them.

    Args:
        data_points: List of DataPoint objects to index. Each DataPoint's metadata must
                     contain an 'index_fields' list specifying which fields to embed.
        vector_engine: Optional pre-created vector engine. Falls back to
                       ``get_vector_engine_async()`` when not supplied.
        semaphore: Optional shared bound on concurrent batch writes.

    Returns:
        The original data_points list.
    """
    vector_engine = vector_engine or await get_vector_engine_async()

    points_by_collection: dict[tuple[str, str], dict[UUID, DataPoint]] = {}
    for data_point in data_points:
        # Skip non-DataPoint objects (e.g. CogneeGraph) that may be
        # passed through the memify pipeline without metadata.
        metadata = getattr(data_point, "metadata", None)
        if not metadata:
            continue

        type_name = type(data_point).__name__
        for field_name in metadata.get("index_fields") or []:
            if getattr(data_point, field_name, None) is None:
                continue
            key = (type_name, field_name)
            points = points_by_collection.get(key)
            if points is None:
                await vector_engine.create_vector_index(type_name, field_name)
                points = points_by_collection[key] = {}
            point = _for_field(data_point, field_name)
            existing = points.get(point.id)
            if existing is not None:
                # Identity normalization can give distinct edge texts the same ID.
                # Collapse before batching: adapter-local deduplication cannot keep
                # duplicates in separate batches from being embedded/written twice.
                # Match the adapters' last-payload, union-of-tags conflict behavior.
                tags = list(
                    dict.fromkeys((existing.belongs_to_set or []) + (point.belongs_to_set or []))
                )
                if tags and tags != point.belongs_to_set:
                    point = point.model_copy(update={"belongs_to_set": tags})
            points[point.id] = point

    if semaphore is None:
        semaphore = indexing_semaphore(vector_engine)
    rows_per_write = write_batch_size()

    async def _index_batch(type_name, field_name, batch):
        async with semaphore:
            await vector_engine.index_data_points(type_name, field_name, batch)

    tasks = []
    for (type_name, field_name), points_by_id in points_by_collection.items():
        points = list(points_by_id.values())
        tasks.extend(
            asyncio.create_task(
                _index_batch(type_name, field_name, points[start : start + rows_per_write])
            )
            for start in range(0, len(points), rows_per_write)
        )
    # A failed batch must not leave sibling writes running after this call returns.
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome

    return data_points
