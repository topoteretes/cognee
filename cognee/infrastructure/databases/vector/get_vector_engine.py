from .config import get_vectordb_context_config
from .create_vector_engine import acreate_vector_engine


async def get_vector_engine():
    """Factory function to get the appropriate vector client.

    Resolves the engine eagerly and returns the live adapter (a leased proxy
    from ``closing_lru_cache``) directly. Uses the async creation path
    (``acreate_vector_engine``) so that if the engine for this config key is
    currently closing, we await that close before constructing a new one.

    SPIKE NOTE (async-engine-resolution): this previously was SYNC and returned
    a ``_VectorEngineHandle`` whose ``__getattr__`` re-resolved the engine on
    every attribute access. Every call site must now ``await get_vector_engine()``.
    Trade-off: callers that store the result across a cache-invalidating op
    (prune / delete / context exit) must re-call instead of relying on the old
    transparent re-resolution.
    """
    return await acreate_vector_engine(**get_vectordb_context_config())
