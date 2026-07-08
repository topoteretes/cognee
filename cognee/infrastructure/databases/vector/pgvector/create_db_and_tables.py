from sqlalchemy import text

from ..get_vector_engine import get_vector_engine_async, get_vectordb_context_config
from cognee.context_global_variables import backend_access_control_enabled


async def create_db_and_tables():
    # Get appropriate vector db configuration based on current async context
    vector_config = get_vectordb_context_config()
    vector_engine = await get_vector_engine_async()

    if vector_config["vector_db_provider"] == "pgvector" and not backend_access_control_enabled():
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
