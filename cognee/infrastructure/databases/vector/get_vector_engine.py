from .config import get_vectordb_context_config
from .create_vector_engine import create_vector_engine


def get_vector_engine():
    # Get appropriate vector db configuration based on current async context
    return create_vector_engine(**get_vectordb_context_config())
