from ..get_vector_engine import get_vector_engine, get_vectordb_config
from sqlalchemy import text
from cognee.context_global_variables import vector_db_config as context_vector_db_config


async def create_db_and_tables():
    if context_vector_db_config.get():
        vector_config = context_vector_db_config.get()
    else:
        vector_config = get_vectordb_config().to_dict()
    vector_engine = get_vector_engine()

    if vector_config["vector_db_provider"] == "pgvector":
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
