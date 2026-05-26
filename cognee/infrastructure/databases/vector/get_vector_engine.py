from .config import get_vectordb_context_config
from .create_vector_engine import create_vector_engine


class _VectorEngineHandle:
    """Stable reference to the current vector engine that survives cache invalidation.

    Database engine instances are cached via ``closing_lru_cache``.  Several
    operations invalidate that cache — ``prune_system`` calls ``cache_clear()``,
    ``delete_dataset`` evicts individual entries, and the ``__aexit__`` of
    ``set_database_global_context_variables`` evicts subprocess-mode engines to
    release file locks.  When an entry is evicted the underlying adapter is
    closed, so any direct proxy reference becomes a dead object that raises
    "adapter is closed" on use.

    This handle solves the problem by deferring resolution: every attribute
    access calls ``create_vector_engine(**config)`` which either returns the
    existing cached proxy (fast path) or transparently creates a fresh adapter
    if the old one was evicted (recovery path).  Code that stores the return
    value of ``get_vector_engine()`` — even across ``cognify``, ``search``,
    ``prune``, or ``delete`` calls — always reaches a live adapter without
    needing to re-call ``get_vector_engine()``.
    """

    __slots__ = ("_config",)

    def __init__(self, config: dict):
        object.__setattr__(self, "_config", config)

    def _engine(self):
        return create_vector_engine(**self._config)

    @property
    def __class__(self):
        return self._engine().__class__

    def __getattr__(self, name):
        return getattr(self._engine(), name)

    def __repr__(self):
        return f"<VectorEngineHandle config={self._config!r}>"


def get_vector_engine():
    return _VectorEngineHandle(get_vectordb_context_config())
