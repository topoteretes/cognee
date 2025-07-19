from sqlalchemy import text
from ..get_vector_engine import get_vector_engine, get_vectordb_context_config


async def create_db_and_tables():
    # Get appropriate vector db configuration based on current async context
    vector_config = get_vectordb_context_config()
    vector_engine = get_vector_engine()

    if vector_config["vector_db_provider"] == "pgvector":
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
