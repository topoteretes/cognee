from cognee.infrastructure.databases.utils.get_graph_dataset_database_handler import (
    get_graph_dataset_database_handler,
)
from cognee.infrastructure.databases.utils.get_vector_dataset_database_handler import (
    get_vector_dataset_database_handler,
)
from cognee.modules.users.models.DatasetDatabase import DatasetDatabase


async def resolve_dataset_database_connection_info(
    dataset_database: DatasetDatabase,
) -> DatasetDatabase:
    """
    Resolve the connection info for the given DatasetDatabase instance.
    Resolve both vector and graph database connection info and return the updated DatasetDatabase instance.

    Args:
        dataset_database: DatasetDatabase instance
    Returns:
        DatasetDatabase instance with resolved connection info
    """
    vector_dataset_database_handler = get_vector_dataset_database_handler(dataset_database)
    graph_dataset_database_handler = get_graph_dataset_database_handler(dataset_database)
    dataset_database = await vector_dataset_database_handler[
        "handler_instance"
    ].resolve_dataset_connection_info(dataset_database)
    dataset_database = await graph_dataset_database_handler[
        "handler_instance"
    ].resolve_dataset_connection_info(dataset_database)
    return dataset_database
