from .config import get_vectordb_context_config
from .create_vector_engine import create_vector_engine


async def get_vector_engine():
    """Factory function to get the appropriate vector client.

    SPIKE NOTE (async-engine-resolution): this previously was SYNC and returned
    a ``_VectorEngineHandle`` whose ``__getattr__`` re-resolved the engine via
    ``create_vector_engine(**config)`` on every attribute access. It is now an
    ``async`` function that resolves the engine eagerly and returns the live
    adapter (a leased proxy from ``closing_lru_cache``) directly. Every call
    site must now ``await get_vector_engine()``. Trade-off: callers that store
    the result across a cache-invalidating op (prune / delete / context exit)
    must re-call instead of relying on the old transparent re-resolution.
    """
    return create_vector_engine(**get_vectordb_context_config())
