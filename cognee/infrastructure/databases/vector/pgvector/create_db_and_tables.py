from sqlalchemy import text, URL
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.modules.users.models import DatasetDatabase
from ..get_vector_engine import get_vector_engine, get_vectordb_context_config
from ...vector import get_vectordb_config
from cognee.context_global_variables import backend_access_control_enabled


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

    # Connect to maintenance db in order to create new database
    # Make sure to execute CREATE DATABASE outside of transaction block, and set AUTOCOMMIT isolation level
    connection = await maintenance_engine.connect()
    connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
    await connection.execute(text(f'CREATE DATABASE "{vector_config["vector_db_name"]}";'))

    # Clean up resources
    await connection.close()

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

    connection = await maintenance_engine.connect()
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
    await connection.execute(text(f'DROP DATABASE "{dataset_database.vector_database_name}";'))

    await connection.close()
