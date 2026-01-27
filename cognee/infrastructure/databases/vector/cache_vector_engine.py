import os
from functools import lru_cache

from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import get_embedding_engine
from cognee.infrastructure.databases.vector.redis.RedisVectorAdapter import RedisVectorAdapter

def get_cache_vector_engine(url: str | None = None):

    cache_url = url or os.environ.get("CACHE_VECTOR_URL", "redis://localhost:6379")
    return create_cache_vector_engine(cache_url)


@lru_cache()
def create_cache_vector_engine(url: str):
    embedding_engine = get_embedding_engine()
    return RedisVectorAdapter(url=url, embedding_engine=embedding_engine)
