from .config import get_vectordb_config
from .create_vector_engine import create_vector_engine


def get_vector_engine():
    # Get appropriate vector db configuration based on current async context
    vector_config = get_vectordb_config()

    return create_vector_engine(
        vector_db_provider=vector_config.vector_db_provider,
        vector_db_url=vector_config.vector_db_url,
        vector_db_port=vector_config.vector_db_port,
        vector_db_key=vector_config.vector_db_key,
    )
