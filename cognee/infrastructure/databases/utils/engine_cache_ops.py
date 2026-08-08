"""Shared cache-management facade for the decorated engine factories.

Graph and vector engines expose the same five cache operations — evict,
touch, is-cached, evict-by-database, and its async variant — through one
public instance per engine module: ``graph_engine_cache`` in
``get_graph_engine`` and ``vector_engine_cache`` in ``create_vector_engine``
(e.g. ``graph_engine_cache.evict(force_close=True, **cfg)``). Before this
module each engine file carried its own copy of the plumbing. The split of
responsibilities is deliberate:

* The *knowledge* — how a config dict becomes the factory's exact cache key —
  stays in the engine module and is passed in as ``key_args``. Only the
  factory's module can build that tuple (its own argument normalization);
  duplicating the normalization elsewhere is how keys silently drift.
* The *mechanics* — which cache method implements which operation — live
  here, once.
"""


class EngineCacheOps:
    """Uniform cache-management operations over one decorated engine factory.

    ``factory`` is a ``closing_lru_cache``-decorated function (exposing
    ``cache_evict`` / ``cache_evict_and_close`` / ``cache_touch`` /
    ``cache_contains`` / ``cache_evict_matching`` / ``cache_await_closed``).
    ``key_args`` maps a config dict to the factory's positional cache-key
    tuple. ``database_name_field`` is the cache-key field holding the
    per-dataset database name (a dataset UUID), used by the by-database
    evictions.
    """

    __slots__ = ("_factory", "_key_args", "_database_name_field")

    def __init__(self, factory, key_args, database_name_field: str) -> None:
        self._factory = factory
        self._key_args = key_args
        self._database_name_field = database_name_field

    def evict(self, force_close: bool = False, **kwargs) -> bool:
        """Evict the cached engine entry for this config.

        ``force_close=True`` closes the adapter immediately even while idle
        holders still pin its proxy (they re-resolve on next use) — the
        dataset-queue teardown path, where a worker must exit and release its
        file locks promptly. The default keeps lease semantics: the close
        waits for the last holder. Returns True if the entry existed.
        """
        evict = self._factory.cache_evict_and_close if force_close else self._factory.cache_evict
        return evict(*self._key_args(kwargs))

    def touch(self, **kwargs) -> bool:
        """Refresh the idle timestamp of the cached engine for this config.

        The dataset-queue release path calls this under the idle-TTL
        keep-alive instead of evicting: the engine stays cached (and its
        worker alive) until it has been idle for the TTL. Returns True if the
        entry exists.
        """
        return self._factory.cache_touch(*self._key_args(kwargs))

    def is_cached(self, **kwargs) -> bool:
        """Check whether an engine entry exists for this config without creating."""
        return self._factory.cache_contains(*self._key_args(kwargs))

    def evict_for_database(self, database_name: str) -> int:
        """Evict every cached engine bound to *database_name*.

        The same per-dataset database can be cached under multiple keys: the
        dataset-handler creation key and the pipeline's context-config key
        differ in fields like the file path or the handler name, so key-exact
        :meth:`evict` misses entries and leaves an engine whose backing
        database has been dropped. Per-dataset database names are dataset
        UUIDs, so matching the name against key fields cannot collide with
        other entries. Returns the number of evicted entries.
        """
        if not database_name:
            raise ValueError(f"{self._database_name_field} must be a non-empty database name")
        return self._factory.cache_evict_matching(**{self._database_name_field: database_name})

    async def aevict_for_database(self, database_name: str) -> int:
        """Evict every cached engine bound to *database_name* and wait until
        their IN-FLIGHT closes have completed (workers exited, file locks
        released). Use before removing the database's files so a teardown that
        is already running cannot race the removal.

        A close still deferred behind a live caller proxy (an idle engine
        handle) is NOT waited on — see the ``closing_lru_cache`` module
        docstring. In that case files are removed under an engine that closes
        later; on POSIX the unlinked files stay valid for the holder and the
        eventual close writes to nowhere, which is acceptable for a dataset
        being deleted. Returns the number of evicted entries.
        """
        evicted = self.evict_for_database(database_name)
        await self._factory.cache_await_closed(**{self._database_name_field: database_name})
        return evicted
