import warnings

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

    __slots__ = ("_config", "_pinned")

    # Note: ``get_vector_engine()`` is ``async`` (callers ``await`` it), but the
    # handle's resolution (``_engine``) is intentionally synchronous — there is
    # no async ``acreate``/await-in-flight-close path here. The subprocess vector
    # backend (LanceDB) connects via ``connect_async`` without taking an
    # exclusive on-disk file lock the way Kuzu/Ladybug does, so the
    # create-vs-close lock race that motivates the graph handle's async path does
    # not apply. Like ``_GraphEngineHandle`` this is a *re-resolving* handle: it
    # re-runs ``create_vector_engine(**config)`` on attribute access, so a stored
    # reference transparently recovers a fresh adapter after the cache is
    # invalidated (prune / delete / dataset-context teardown) instead of raising
    # "adapter is closed". The detach-aware pin just avoids re-entering the cache
    # on every access and never keeps an evicted worker pinned alive.

    def __init__(self, config: dict):
        object.__setattr__(self, "_config", config)
        # Pinned leased engine proxy — see ``_GraphEngineHandle`` for the
        # rationale. Dropped + re-resolved once the pin is no longer the live
        # cache entry so prune/delete eviction still recovers a fresh engine.
        object.__setattr__(self, "_pinned", None)

    @staticmethod
    def _pin_is_live(engine) -> bool:
        active = getattr(engine, "_leased_entry_active", None)
        if active is not None:
            try:
                if not active():
                    return False
            except Exception:
                return False
        if getattr(engine, "_permanently_closed", False):
            return False
        return True

    def _engine(self):
        pinned = self._pinned
        if pinned is not None and self._pin_is_live(pinned):
            return pinned
        # Drop the stale pin before re-resolving so its deferred close can start
        # (mirrors ``_GraphEngineHandle._release_stale_pin``).
        if pinned is not None:
            object.__setattr__(self, "_pinned", None)
            del pinned
        engine = create_vector_engine(**self._config)
        object.__setattr__(self, "_pinned", engine)
        return engine

    @property
    def __class__(self):
        return self._engine().__class__

    def __getattr__(self, name):
        return getattr(self._engine(), name)

    def __repr__(self):
        return f"<VectorEngineHandle config={self._config!r}>"


def _new_vector_engine_handle() -> _VectorEngineHandle:
    return _VectorEngineHandle(get_vectordb_context_config())


async def get_vector_engine_async() -> _VectorEngineHandle:
    """Return the vector engine adapter. **This is the canonical way to obtain the
    adapter** — use it (and ``await`` it) everywhere inside cognee and in async
    application code, alongside ``await get_graph_engine()``.

    The ``async`` signature exists to keep a uniform "await the engine getter"
    contract with ``get_graph_engine()`` (which genuinely awaits — Kuzu/Ladybug
    take an exclusive on-disk file lock, so it must wait for an in-flight close
    before reopening). Vector resolution itself is synchronous: LanceDB connects
    via ``connect_async`` without taking that lock, so there is no in-flight-close
    race to await here. Do not "restore symmetry" by adding an ``acreate`` path —
    it was tried and deliberately walked back (see #3708) because it is a no-op
    for LanceDB.
    """
    return _new_vector_engine_handle()


def get_vector_engine() -> _VectorEngineHandle:
    """Deprecated synchronous accessor kept for backward compatibility.

    .. deprecated::
        Use ``await get_vector_engine_async()`` instead. This shim exists so
        existing downstream code that calls ``get_vector_engine()`` without
        ``await`` (e.g. cognee-community adapters pinned to older releases) keeps
        working after vector resolution was moved to the async surface.

    Safe to call synchronously from any context — no running event loop is
    required — because constructing the handle does no async work (see
    ``get_vector_engine_async`` for why vector needs no async resolution).

    Note, however, that the *returned adapter* only works inside a running
    asyncio event loop: its operations (``embed_text``, ``get_connection``,
    ``search``, ...) are coroutines that must be awaited. So this getter is only
    useful to code that will ``await`` the adapter's methods within an event loop.
    """
    warnings.warn(
        "get_vector_engine() is deprecated; use `await get_vector_engine_async()` "
        "instead. The returned adapter's methods are async and must be awaited "
        "inside a running event loop.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_vector_engine_handle()
