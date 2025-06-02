from .config import get_vectordb_config
from .create_vector_engine import create_vector_engine


def get_vector_engine():
    """
    Create and return a vector engine instance.

    Returns:
    --------

        A vector engine instance created from the vector database configuration.
    """
    return create_vector_engine(**get_vectordb_config().to_dict())
