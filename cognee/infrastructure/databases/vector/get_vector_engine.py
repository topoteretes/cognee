from .config import get_vectordb_config
from .embeddings import get_embedding_engine
from .create_vector_engine import create_vector_engine


def get_vector_engine():
    return create_vector_engine(get_embedding_engine(), **get_vectordb_config().to_dict())
