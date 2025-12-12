from sqlalchemy.exc import OperationalError

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.cache import delete_cache
from cognee.modules.users.models import DatasetDatabase
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def prune_graph_databases():
    async def _prune_graph_db(dataset_database: DatasetDatabase) -> dict:
        from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
            supported_dataset_database_handlers,
        )

        handler = supported_dataset_database_handlers[
            dataset_database.graph_dataset_database_handler
        ]
        return await handler["handler_instance"].delete_dataset(dataset_database)

    db_engine = get_relational_engine()
    try:
        data = await db_engine.get_all_data_from_table("dataset_database")
        # Go through each dataset database and delete the graph database
        for data_item in data:
            await _prune_graph_db(data_item)
    except (OperationalError, EntityNotFoundError) as e:
        logger.debug(
            "Skipping pruning of graph DB. Error when accessing dataset_database table: %s",
            e,
        )
        return


async def prune_vector_databases():
    async def _prune_vector_db(dataset_database: DatasetDatabase) -> dict:
        from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
            supported_dataset_database_handlers,
        )

        handler = supported_dataset_database_handlers[
            dataset_database.vector_dataset_database_handler
        ]
        return await handler["handler_instance"].delete_dataset(dataset_database)

    db_engine = get_relational_engine()
    try:
        data = await db_engine.get_all_data_from_table("dataset_database")
        # Go through each dataset database and delete the vector database
        for data_item in data:
            await _prune_vector_db(data_item)
    except (OperationalError, EntityNotFoundError) as e:
        logger.debug(
            "Skipping pruning of vector DB. Error when accessing dataset_database table: %s",
            e,
        )
        return


async def prune_system(graph=True, vector=True, metadata=True, cache=True):
    # Note: prune system should not be available through the API, it has no permission checks and will
    #       delete all graph and vector databases if called. It should only be used in development or testing environments.
    if graph and not backend_access_control_enabled():
        graph_engine = await get_graph_engine()
        await graph_engine.delete_graph()
    elif graph and backend_access_control_enabled():
        await prune_graph_databases()

    if vector and not backend_access_control_enabled():
        vector_engine = get_vector_engine()
        await vector_engine.prune()
    elif vector and backend_access_control_enabled():
        await prune_vector_databases()

    if metadata:
        db_engine = get_relational_engine()
        await db_engine.delete_database()

    if cache:
        await delete_cache()
