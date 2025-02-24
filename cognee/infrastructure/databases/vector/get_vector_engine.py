from .config import get_vectordb_config
from .create_vector_engine import create_vector_engine


def get_vector_engine():
    return create_vector_engine(**get_vectordb_config().to_dict())
