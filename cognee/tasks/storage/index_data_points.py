import asyncio

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_context_config
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger("index_data_points")


# Rows per lexical indexing call: one statement on Postgres, one add on
# LanceDB. Independent of the embedding batch size, which is sized for the
# embedding API.
LEXICAL_BATCH_SIZE = 500


async def index_data_points(data_points: list[DataPoint], vector_engine=None):
    """Index data points in the vector engine by creating embeddings for specified fields.

    Data points are indexed in batches of the embedding engine's batch_size. A semaphore
    bounds how many batches run concurrently so that at most
    ``embedding_max_concurrent_data_points`` data points (default 150, env
    ``EMBEDDING_MAX_CONCURRENT_DATA_POINTS``) are in flight at once:
    ``max(1, embedding_max_concurrent_data_points // batch_size)`` concurrent batches.

    Fields named in ``metadata["lexical_fields"]`` go to the collection's keyword (BM25)
    index as well, after the embeddings, on backends with ``supports_lexical_search``;
    a field in ``lexical_fields`` but not in ``index_fields`` is stored as rows without
    a vector. Other backends skip the lexical step (the lexical retrievers then score
    the graph nodes in memory).

    Args:
        data_points: List of DataPoint objects to index. Each DataPoint's metadata must
                     contain an 'index_fields' list specifying which fields to embed.
        vector_engine: Optional pre-created vector engine. Falls back to
                       ``get_vector_engine_async()`` when not supplied.

    Returns:
        The original data_points list.
    """
    data_points_by_type = {}
    # (type, field) -> (owns_rows, points) for the lexical index.
    lexical_points_by_field: dict[tuple[str, str], tuple[bool, list[DataPoint]]] = {}

    vector_engine = vector_engine or await get_vector_engine_async()
    lexical_supported = getattr(vector_engine, "supports_lexical_search", False) is True

    for data_point in data_points:
        # Skip non-DataPoint objects (e.g. CogneeGraph) that may be
        # passed through the memify pipeline without metadata.
        if not hasattr(data_point, "metadata") or not data_point.metadata:
            continue

        data_point_type = type(data_point)
        type_name = data_point_type.__name__
        index_fields = data_point.metadata["index_fields"]

        for field_name in index_fields:
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

        if not lexical_supported:
            continue
        for field_name in data_point.metadata.get("lexical_fields") or []:
            if getattr(data_point, field_name, None) is None:
                continue
            owns_rows = field_name not in index_fields
            key = (type_name, field_name)
            if key not in lexical_points_by_field:
                await vector_engine.create_lexical_index(type_name, field_name)
                lexical_points_by_field[key] = (owns_rows, [])
            lexical_points_by_field[key][1].append(data_point)

    batch_size = vector_engine.embedding_engine.get_batch_size()
    max_concurrent_data_points = get_embedding_context_config().embedding_max_concurrent_data_points
    semaphore = asyncio.Semaphore(max(1, max_concurrent_data_points // batch_size))

    async def _index_batch(type_name, field_name, batch):
        async with semaphore:
            await vector_engine.index_data_points(type_name, field_name, batch)

    tasks = []
    for type_name, fields in data_points_by_type.items():
        for field_name, points in fields.items():
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                tasks.append(asyncio.create_task(_index_batch(type_name, field_name, batch)))

    await asyncio.gather(*tasks)

    # After the embeddings: a field that is both embedded and lexical must find
    # its rows written (owns_rows=False keeps the vector); a lexical-only field
    # writes its own rows here.
    for (type_name, field_name), (owns_rows, points) in lexical_points_by_field.items():
        for i in range(0, len(points), LEXICAL_BATCH_SIZE):
            await vector_engine.index_lexical_data_points(
                type_name, field_name, points[i : i + LEXICAL_BATCH_SIZE], owns_rows=owns_rows
            )

    return data_points
