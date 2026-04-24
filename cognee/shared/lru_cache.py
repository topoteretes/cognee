"""
Shared cache size setting for Cognee database adapters.

Set DATABASE_MAX_LRU_CACHE_SIZE in the environment to control the maxsize of the
lru_cache used by the graph, vector, and relational DB engine factories (default: 128).
"""

import os

DATABASE_MAX_LRU_CACHE_SIZE: int = int(os.getenv("DATABASE_MAX_LRU_CACHE_SIZE", "128"))
