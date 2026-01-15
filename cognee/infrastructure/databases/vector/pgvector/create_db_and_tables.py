from sqlalchemy import text
from ..get_vector_engine import get_vector_engine, get_vectordb_context_config


async def create_db_and_tables():
    # Get appropriate vector db configuration based on current async context
    vector_config = get_vectordb_context_config()
    vector_engine = get_vector_engine()

    if vector_config["vector_db_provider"] == "pgvector":
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


async def create_pg_database(vector_config):
    """
    Create the necessary Postgres database, and the PGVector extension on it.
    This is defined separately because the creation needs the latest vector config,
    which is not yet saved in the vector config context variable.

    TODO: We can maybe merge this with create_db_and_tables(), but it seemed simpler to separate them for now
    """
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
    from sqlalchemy import text

    pg_relational_engine = create_relational_engine(
        db_path="",
        db_host=vector_config["vector_db_url"],
        db_name=vector_config["vector_db_name"],
        db_port=vector_config["vector_db_port"],
        db_username=vector_config["vector_db_username"],
        db_password=vector_config["vector_db_password"],
        db_provider="postgres",
    )
    await pg_relational_engine.create_database()

    vector_engine = create_vector_engine(**vector_config)
    async with vector_engine.engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
