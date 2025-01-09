from ..get_vector_engine import get_vector_engine, get_vectordb_config
from sqlalchemy import text


async def create_db_and_tables():
    vector_config = get_vectordb_config()
    vector_engine = get_vector_engine()

    if vector_config.vector_db_provider == "pgvector":
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
