from sqlalchemy.exc import OperationalError

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.get_graph_engine import (
    _create_graph_engine,
    get_graph_engine,
)
from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils import (
    get_graph_dataset_database_handler,
    get_vector_dataset_database_handler,
)
from cognee.shared.cache import delete_cache
from cognee.infrastructure.databases.cache import (
    get_cache_config,
    get_cache_engine,
)
from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
from cognee.modules.users.models import DatasetDatabase
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def prune_graph_databases():
    db_engine = get_relational_engine()
    try:
        dataset_databases = await db_engine.get_all_data_from_table("dataset_database")
        # Go through each dataset database and delete the graph database
        for dataset_database in dataset_databases:
            handler = get_graph_dataset_database_handler(dataset_database)
            await handler["handler_instance"].delete_dataset(dataset_database)
    except (OperationalError, EntityNotFoundError) as e:
        logger.debug(
            "Skipping pruning of graph DB. Error when accessing dataset_database table: %s",
            e,
        )
        return


async def prune_vector_databases():
    db_engine = get_relational_engine()
    try:
        dataset_databases = await db_engine.get_all_data_from_table("dataset_database")
        # Go through each dataset database and delete the vector database
        for dataset_database in dataset_databases:
            handler = get_vector_dataset_database_handler(dataset_database)
            await handler["handler_instance"].delete_dataset(dataset_database)
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
        # delete_graph() only does a transient drop and leaves any subprocess
        # worker alive. Close the engine here so the worker is terminated and
        # its inherited OS handles (e.g. the cognee_db relational file on
        # Windows, where children inherit handles at multiprocessing 'spawn'
        # time) are released BEFORE delete_database() removes those files.
        # Mirrors the awaited per-dataset teardown in
        # LadybugDatasetDatabaseHandler.delete_dataset (delete_graph ->
        # evict -> await close()). Without this await the close() scheduled by
        # cache_clear() below is fire-and-forget and races the file delete,
        # producing PermissionError [WinError 32] on Windows.
        # Only the ladybug/kuzu adapter exposes close(); neo4j/networkx don't,
        # so guard with hasattr to stay provider-agnostic.
        if hasattr(graph_engine, "close"):
            await graph_engine.close()  # type: ignore[attr-defined]
    elif graph and backend_access_control_enabled():
        await prune_graph_databases()

    if vector and not backend_access_control_enabled():
        vector_engine = get_vector_engine()
        await vector_engine.prune()
        # Same rationale as the graph engine above: deterministically close
        # the vector engine (terminating any subprocess worker and releasing
        # inherited handles) before metadata deletion, instead of relying on
        # the fire-and-forget close scheduled by cache_clear(). Only the
        # LanceDB adapter exposes close(); ChromaDB/PGVector don't, so guard
        # with hasattr to stay provider-agnostic.
        if hasattr(vector_engine, "close"):
            await vector_engine.close()
    elif vector and backend_access_control_enabled():
        await prune_vector_databases()

    if graph:
        _create_graph_engine.cache_clear()

    if vector:
        _create_vector_engine.cache_clear()

    if metadata:
        db_engine = get_relational_engine()
        await db_engine.delete_database()

    if cache:
        await delete_cache()
        cache_config = get_cache_config()
        if cache_config.caching or cache_config.usage_logging:
            create_cache_engine.cache_clear()
            cache_engine = get_cache_engine()
            if cache_engine:
                await cache_engine.prune()
