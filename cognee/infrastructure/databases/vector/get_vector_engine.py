from .config import get_vectordb_context_config
from .create_vector_engine import acreate_vector_engine


async def get_vector_engine():
    """Resolve the vector engine for the current context and return the live
    adapter (a leased proxy from ``closing_lru_cache``).

    Resolution is asynchronous and goes through :func:`acreate_vector_engine`
    so engine *creation* can ``await`` an in-flight close of the same cache key
    before constructing (see :func:`acreate_vector_engine` for the rationale —
    the same async-factory reasoning as the graph side keeps the two engines
    symmetric, even though LanceDB itself takes no exclusive file lock).

    ``get_vector_engine()`` is ``async`` — every call site must ``await`` it.
    The returned adapter is a live reference for the *current* async scope; do
    not stash it and reuse it across a cache-invalidating operation (prune /
    delete / per-dataset context exit) — call ``get_vector_engine()`` again.
    """
    return await acreate_vector_engine(**get_vectordb_context_config())
