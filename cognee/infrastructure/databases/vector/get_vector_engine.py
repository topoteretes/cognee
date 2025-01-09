from .config import get_vectordb_config
from .embeddings import get_embedding_engine
from .create_vector_engine import create_vector_engine
from functools import lru_cache


@lru_cache
def get_vector_engine():
    return create_vector_engine(get_vectordb_config().to_dict(), get_embedding_engine())
