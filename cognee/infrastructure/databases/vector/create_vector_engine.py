from .embeddings import get_embedding_engine
from .supported_adapters import supported_adapters

from functools import lru_cache


@lru_cache
def create_vector_engine(
    vector_db_url: str,
    vector_db_port: str,
    vector_db_key: str,
    vector_db_provider: str,
):
    embedding_engine = get_embedding_engine()

    vector_db_adapter = supported_adapters.get(vector_db_provider, None)

    return vector_db_adapter(url=vector_db_url, api_key=vector_db_key, embedding_engine=embedding_engine)
