from sqlalchemy import text, URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.modules.users.models import DatasetDatabase
from ..get_vector_engine import get_vector_engine, get_vectordb_context_config
from ...vector import get_vectordb_config
from cognee.context_global_variables import backend_access_control_enabled
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


async def _dispose_cached_pgvector_engine(vector_config: dict):
    """
    Dispose cached PGVector engine pool for a dataset database if it exists.
    This prevents reusing connections that were closed when the dataset database was dropped.
    """

    from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine

    try:
        vector_engine = create_vector_engine(**vector_config)
        await vector_engine.engine.dispose()
    except (TypeError, KeyError, AttributeError) as error:
        logger.warning(
            "Unable to dispose cached PGVector engine due to invalid config for db '%s': %s",
            vector_config.get("vector_db_name", "<unknown>"),
            error,
        )
    except (SQLAlchemyError, RuntimeError) as error:
        logger.warning(
            "Failed to dispose cached PGVector engine for db '%s': %s",
            vector_config.get("vector_db_name", "<unknown>"),
            error,
        )


async def create_db_and_tables():
    # Get appropriate vector db configuration based on current async context
    vector_config = get_vectordb_context_config()
    vector_engine = get_vector_engine()

    if vector_config["vector_db_provider"] == "pgvector" and not backend_access_control_enabled():
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


async def create_pg_database(vector_config):
    """
    Create the necessary Postgres database, and the PGVector extension on it.
    This is defined separately because the creation needs the latest vector config,
    which is not yet saved in the vector config context variable.
    """

    from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine

    # Create a maintenance engine, used when creating new postgres databases.
    # Database named "postgres" should always exist. We need this since the SQLAlchemy
    # engine cannot directly execute queries without first connecting to a database.
    maintenance_db_name = "postgres"
    maintenance_db_url = URL.create(
        "postgresql+asyncpg",
        username=vector_config["vector_db_username"],
        password=vector_config["vector_db_password"],
        host=vector_config["vector_db_host"],
        port=int(vector_config["vector_db_port"]),
        database=maintenance_db_name,
    )
    maintenance_engine = create_async_engine(maintenance_db_url)
    try:
        # Connect to maintenance db in order to create new database
        # Make sure to execute CREATE DATABASE outside of transaction block, and set AUTOCOMMIT isolation level
        connection = await maintenance_engine.connect()
        try:
            connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
            await connection.execute(text(f'CREATE DATABASE "{vector_config["vector_db_name"]}";'))
        finally:
            await connection.close()
    finally:
        await maintenance_engine.dispose()

    # If the same dataset database name was previously dropped/recreated, ensure we don't reuse stale pooled connections.
    await _dispose_cached_pgvector_engine(vector_config)
    vector_engine = create_vector_engine(**vector_config)
    async with vector_engine.engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


async def delete_pg_database(dataset_database: DatasetDatabase):
    """
    Delete the Postgres database that was created for the PGVector extension,
    in case of using PGVector with multi-user mode.
    """

    vector_config = get_vectordb_config()
    # Create a maintenance engine, used when creating new postgres databases.
    # Database named "postgres" should always exist. We need this since the SQLAlchemy
    # engine cannot drop a database to which it is connected.
    maintenance_db_name = "postgres"
    maintenance_db_url = URL.create(
        "postgresql+asyncpg",
        username=vector_config.vector_db_username,
        password=vector_config.vector_db_password,
        host=vector_config.vector_db_host,
        port=int(vector_config.vector_db_port),
        database=maintenance_db_name,
    )
    maintenance_engine = create_async_engine(maintenance_db_url)

    # Dispose cached dataset engine first so its pool doesn't keep stale handles for this DB.
    await _dispose_cached_pgvector_engine(
        {
            "vector_db_provider": vector_config.vector_db_provider,
            "vector_db_url": vector_config.vector_db_url,
            "vector_db_name": dataset_database.vector_database_name,
            "vector_db_port": dataset_database.vector_database_connection_info.get(
                "port", vector_config.vector_db_port
            ),
            "vector_db_key": "",
            "vector_db_username": dataset_database.vector_database_connection_info.get(
                "username", vector_config.vector_db_username
            ),
            "vector_db_password": dataset_database.vector_database_connection_info.get(
                "password", vector_config.vector_db_password
            ),
            "vector_db_host": dataset_database.vector_database_connection_info.get(
                "host", vector_config.vector_db_host
            ),
            "vector_dataset_database_handler": "pgvector",
        }
    )

    try:
        connection = await maintenance_engine.connect()
        try:
            connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
            # We first have to kill all active sessions on the database, then delete it
            await connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid()"
                ),
                {"db": dataset_database.vector_database_name},
            )
            await connection.execute(
                text(f'DROP DATABASE "{dataset_database.vector_database_name}";')
            )
        finally:
            await connection.close()
    finally:
        await maintenance_engine.dispose()
