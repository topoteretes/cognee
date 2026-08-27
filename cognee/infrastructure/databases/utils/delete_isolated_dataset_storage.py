from cognee.infrastructure.databases.utils.get_graph_dataset_database_handler import (
    get_graph_dataset_database_handler,
)
from cognee.infrastructure.databases.utils.get_vector_dataset_database_handler import (
    get_vector_dataset_database_handler,
)
from cognee.modules.users.models.DatasetDatabase import DatasetDatabase


async def delete_isolated_dataset_storage(dataset_database: DatasetDatabase) -> None:
    """Drop the physical graph and vector databases for one dataset.

    Same two handler calls ``cognee.modules.data.methods.delete_dataset`` makes
    before it removes the ``Dataset`` row, factored out so a caller that must
    keep the ``Dataset`` row, its ``Data`` rows, and this ``DatasetDatabase``
    registry row (e.g. a memory-only reset) can reuse the exact primitive
    without any of that relational cleanup.

    Only call this for a dataset that is confirmed to own an isolated
    per-dataset graph/vector database pair — this function does not check;
    callers resolve ``dataset_database`` via
    ``get_existing_dataset_database`` first.
    """
    graph_handler = get_graph_dataset_database_handler(dataset_database)
    vector_handler = get_vector_dataset_database_handler(dataset_database)
    await graph_handler["handler_instance"].delete_dataset(dataset_database)
    await vector_handler["handler_instance"].delete_dataset(dataset_database)
