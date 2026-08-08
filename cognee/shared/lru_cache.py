"""
Shared cache size setting for Cognee database adapters.

Set DATABASE_MAX_LRU_CACHE_SIZE in the environment to control the maxsize of the
closing LRU cache used by the graph, vector, and relational DB engine factories
(default: 6). It is also the default for DATASET_QUEUE_MAX_CONCURRENT, keeping
admitted datasets and cache capacity in step. Engines of datasets currently
holding a dataset-queue slot are pinned against capacity eviction, so the cache
may briefly exceed this size under full concurrency — bounded by the queue's
slot count.
"""

import os

DATABASE_MAX_LRU_CACHE_SIZE: int = int(os.getenv("DATABASE_MAX_LRU_CACHE_SIZE", "6"))
