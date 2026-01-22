from sqlalchemy import text

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
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
    from sqlalchemy import text

    pg_relational_engine = create_relational_engine(
        db_path="",
        db_host=vector_config["vector_db_host"],
        db_name=vector_config["vector_db_name"],
        db_port=vector_config["vector_db_port"],
        db_username=vector_config["vector_db_username"],
        db_password=vector_config["vector_db_password"],
        db_provider="postgres",
    )
    # await pg_relational_engine.create_database()

    # Connect to maintenance db in order to create new database
    # Make sure to execute CREATE DATABASE outside of transaction block, and set AUTOCOMMIT isolation level
    connection = await pg_relational_engine.maintenance_engine.connect()
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

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    pg_relational_engine = create_relational_engine(
        db_path="",
        db_host=dataset_database.vector_database_connection_info["host"],
        db_name=dataset_database.vector_database_name,
        db_port=dataset_database.vector_database_connection_info["port"],
        db_username=vector_config.vector_db_username,
        db_password=vector_config.vector_db_password,
        db_provider="postgres",
    )

    connection = await pg_relational_engine.maintenance_engine.connect()
    connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
    await pg_relational_engine.engine.dispose(close=True)
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
