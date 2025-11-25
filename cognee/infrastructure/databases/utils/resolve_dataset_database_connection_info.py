from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.users.models.DatasetDatabase import DatasetDatabase


async def _get_vector_db_connection_info(dataset_database: DatasetDatabase) -> DatasetDatabase:
    vector_config = get_vectordb_config()

    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[vector_config.vector_dataset_database_handler]
    return await handler["handler_instance"].resolve_dataset_connection_info(dataset_database)


async def _get_graph_db_connection_info(dataset_database: DatasetDatabase) -> DatasetDatabase:
    graph_config = get_graph_config()

    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[graph_config.graph_dataset_database_handler]
    return await handler["handler_instance"].resolve_dataset_connection_info(dataset_database)


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
    dataset_database = await _get_vector_db_connection_info(dataset_database)
    dataset_database = await _get_graph_db_connection_info(dataset_database)
    return dataset_database
