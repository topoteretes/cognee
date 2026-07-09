"""Factory function to get the appropriate graph client based on the graph type."""

import inspect
import os
from numbers import Number

from cognee.infrastructure.databases.utils.closing_lru_cache import closing_lru_cache
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE
from cognee.shared.logging_utils import get_logger

from .kuzu.adapter import DEFAULT_KUZU_BUFFER_POOL_SIZE, DEFAULT_KUZU_MAX_DB_SIZE
from .config import get_graph_context_config
from .graph_db_interface import GraphDBInterface
from .supported_databases import supported_databases

logger = get_logger("GraphEngine")


def _normalize_graph_database_provider(provider: str) -> str:
    return provider.lower() if isinstance(provider, str) else provider


def _get_create_graph_engine_optional_defaults() -> dict:
    """Return default values for optional create_graph_engine parameters."""
    signature = inspect.signature(create_graph_engine)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _normalize_optional_create_graph_engine_params(params: dict) -> dict:
    """
    Normalize optional create_graph_engine parameters:
    - replace None with the function defaults
    - convert numeric graph_database_port values to string
    """
    defaults = _get_create_graph_engine_optional_defaults()
    normalized = dict(params)

    for key, default_value in defaults.items():
        if normalized.get(key) is None:
            normalized[key] = default_value

    if isinstance(normalized.get("graph_database_port"), Number) and not isinstance(
        normalized["graph_database_port"], bool
    ):
        normalized["graph_database_port"] = str(normalized["graph_database_port"])

    if not normalized.get("graph_dataset_database_handler"):
        normalized["graph_dataset_database_handler"] = os.getenv(
            "GRAPH_DATASET_DATABASE_HANDLER", "ladybug"
        )

    return normalized


class _GraphEngineHandle:
    """Stable reference to the current graph engine that survives cache invalidation.

    Database engine instances are cached via ``closing_lru_cache``.  Several
    operations invalidate that cache — ``prune_system`` calls ``cache_clear()``,
    ``delete_dataset`` evicts individual entries, and the ``__aexit__`` of
    ``set_database_global_context_variables`` evicts subprocess-mode engines to
    release file locks.  When an entry is evicted the underlying adapter is
    closed, so any direct proxy reference becomes a dead object that raises
    "adapter is closed" on use.

    This handle solves the problem by deferring resolution: every attribute
    access calls ``create_graph_engine(**config)`` which either returns the
    existing cached proxy (fast path) or transparently creates a fresh adapter
    if the old one was evicted (recovery path).  Code that stores the return
    value of ``get_graph_engine()`` — even across ``cognify``, ``search``,
    ``prune``, or ``delete`` calls — always reaches a live adapter without
    needing to re-call ``get_graph_engine()``.

    For adapters that expose ``initialize()`` (Postgres, Neo4j), the handle
    tracks which engine proxy was last initialized and re-runs the idempotent
    schema setup when the underlying engine changes.

    Known limitation (subprocess + exclusive file lock, e.g. Ladybug): the cache
    leases a single shared proxy per entry, so two concurrently-held handles for
    the same DB path pin the *same* proxy. If that entry is evicted while one
    handle keeps holding it and never re-resolves (a long-lived, idle second
    handle), the old worker's close stays deferred — it does not release the
    file lock, and a fresh engine for the same path falls back to the worker's
    open-retry (``SUBPROCESS_OPEN_LOCK_RETRIES``) rather than the deterministic
    await-the-close path. This is inherent to "one exclusive lock per path with
    concurrent live holders" and is narrow in practice: the primary multi-tenant
    teardown path (``dataset_queue._teardown_subprocess_engines``) ``await``s
    ``engine.close()`` to completion before any re-creation, and a handle that is
    accessed again or garbage-collected drops its stale pin and converges. A
    permanently-idle second handle is the only unrescued case.
    """

    __slots__ = ("_config", "_last_initialized_id", "_pinned")

    def __init__(self, config: dict):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_last_initialized_id", None)
        # Pinned leased engine proxy. Holding it avoids re-entering the cache on
        # every attribute access (and the create-vs-close race that re-entry
        # caused). It is dropped + re-resolved once the pin is no longer the
        # live cache entry (see ``_pin_is_live``) so prune/delete eviction still
        # recovers a fresh engine instead of keeping an evicted DB worker alive.
        object.__setattr__(self, "_pinned", None)

    @staticmethod
    def _pin_is_live(engine) -> bool:
        """Whether a pinned engine is still the live cached value and safe to
        reuse. A leased proxy whose entry was evicted must be released so its
        deferred close can run (otherwise a pinned handle would keep an evicted
        Ladybug worker alive holding the file lock, blocking a new worker)."""
        active = getattr(engine, "_leased_entry_active", None)
        if active is not None:
            try:
                if not active():
                    return False
            except Exception:
                return False
        # Subprocess adapters latch ``_permanently_closed`` on close.
        if getattr(engine, "_permanently_closed", False):
            return False
        return True

    def _release_stale_pin(self, pinned) -> None:
        """Drop the stale pinned proxy BEFORE re-resolving a replacement.

        Critical for the lock race: the pinned proxy is (typically) the last
        reference keeping an evicted adapter alive. Releasing it lets the
        deferred close start — and a subprocess adapter's close runs off-loop,
        releasing the on-disk file lock — *before* a new worker opens the same
        path. Holding the pin across the re-resolution would keep the old worker
        alive and the new one would fail to take the lock.
        """
        object.__setattr__(self, "_pinned", None)
        del pinned

    def _engine(self):
        """Synchronous resolution used on the hot attribute-access path. Reuses
        the pin when live; otherwise drops it and re-resolves through the (sync)
        cache. A mid-flow re-resolution can't await an in-flight close — the
        off-loop close + worker open-retry backstop cover that residual race."""
        pinned = self._pinned
        if pinned is not None and self._pin_is_live(pinned):
            return pinned
        if pinned is not None:
            self._release_stale_pin(pinned)
            pinned = None
        engine = create_graph_engine(**self._config)
        object.__setattr__(self, "_pinned", engine)
        return engine

    async def _aengine(self):
        """Async resolution used at initialization. Goes through the cache's
        async acquisition path so it waits for any in-flight close of the same
        key before constructing a new engine + pinning it."""
        pinned = self._pinned
        if pinned is not None and self._pin_is_live(pinned):
            return pinned
        if pinned is not None:
            self._release_stale_pin(pinned)
            pinned = None
        engine = await acreate_graph_engine(**self._config)
        object.__setattr__(self, "_pinned", engine)
        return engine

    async def _ensure_initialized(self):
        engine = await self._aengine()
        engine_id = id(engine)
        if engine_id != self._last_initialized_id and hasattr(engine, "initialize"):
            await engine.initialize()
        object.__setattr__(self, "_last_initialized_id", engine_id)

    @property
    def __class__(self):
        return self._engine().__class__

    def __getattr__(self, name):
        return getattr(self._engine(), name)

    def __repr__(self):
        return f"<GraphEngineHandle config={self._config!r}>"


async def get_graph_engine() -> GraphDBInterface:
    """Factory function to get the appropriate graph client based on the graph type."""
    config = get_graph_context_config()
    handle = _GraphEngineHandle(config)
    await handle._ensure_initialized()
    return handle


def _make_pghybrid_adapter():
    """Build the uncached Postgres hybrid adapter used when
    ``USE_UNIFIED_PROVIDER=pghybrid``. Not cached — the caller owns it, matching
    the original inline behavior."""
    from .postgres.adapter import PostgresAdapter
    from cognee.infrastructure.databases.relational.get_relational_engine import (
        get_relational_engine,
    )

    return PostgresAdapter(connection_string=get_relational_engine().db_uri)


def _resolve_graph_engine_args(params: dict) -> tuple:
    """Normalize the engine parameters and return the positional argument tuple
    passed to ``_create_graph_engine``.

    Shared by the sync (:func:`create_graph_engine`) and async
    (:func:`acreate_graph_engine`) entry points so both produce the *identical*
    cache key (the positional tuple) — and so it matches the key built by
    ``evict_graph_engine`` / ``is_graph_engine_cached``.
    """
    normalized = _normalize_optional_create_graph_engine_params(params)
    return (
        _normalize_graph_database_provider(params.get("graph_database_provider")),
        params.get("graph_file_path"),
        normalized["graph_database_url"],
        normalized["graph_database_name"],
        normalized["graph_database_username"],
        normalized["graph_database_password"],
        normalized["graph_database_host"],
        normalized["graph_database_allow_anonymous"],
        normalized["graph_database_port"],
        normalized["graph_database_key"],
        normalized["graph_dataset_database_handler"],
        normalized["graph_database_subprocess_enabled"],
        normalized["kuzu_num_threads"],
        normalized["kuzu_buffer_pool_size"],
        normalized["kuzu_max_db_size"],
    )


def create_graph_engine(
    graph_database_provider,
    graph_file_path,
    graph_database_url="",
    graph_database_name="",
    graph_database_username="",
    graph_database_password="",
    graph_database_host="",
    graph_database_allow_anonymous=False,
    graph_database_port="",
    graph_database_key="",
    graph_dataset_database_handler="",
    graph_database_subprocess_enabled=False,
    kuzu_num_threads=0,
    kuzu_buffer_pool_size=DEFAULT_KUZU_BUFFER_POOL_SIZE,
    kuzu_max_db_size=DEFAULT_KUZU_MAX_DB_SIZE,
):
    """
    Wrapper function to call create graph engine with caching.
    For a detailed description, see _create_graph_engine.
    """
    # Check USE_UNIFIED_PROVIDER outside the cache so it's always re-read
    if os.environ.get("USE_UNIFIED_PROVIDER", "") == "pghybrid":
        return _make_pghybrid_adapter()

    return _create_graph_engine(*_resolve_graph_engine_args(locals()))


async def acreate_graph_engine(**kwargs):
    """Async counterpart of :func:`create_graph_engine` that waits for any
    in-flight close of the same cache key before constructing a new engine.

    Used by ``get_graph_engine``'s handle at initialization so a freshly evicted
    subprocess engine's worker has fully exited (releasing its file lock) before
    a new worker opens the same DB path.
    """
    if os.environ.get("USE_UNIFIED_PROVIDER", "") == "pghybrid":
        return _make_pghybrid_adapter()

    return await _create_graph_engine.acall(*_resolve_graph_engine_args(kwargs))


def evict_graph_engine(**kwargs) -> bool:
    """Evict a cached graph engine entry created via ``create_graph_engine``.

    Mirrors ``create_graph_engine``'s normalization so the cache key
    matches. Used by per-dataset deletion paths to drop the leased
    adapter (and trigger its ``close()``) without disturbing the rest
    of the cache.

    Returns True if the entry existed.
    """
    normalized = _normalize_optional_create_graph_engine_params(kwargs)
    provider = _normalize_graph_database_provider(kwargs.get("graph_database_provider"))
    return _create_graph_engine.cache_evict(
        provider,
        kwargs.get("graph_file_path"),
        normalized["graph_database_url"],
        normalized["graph_database_name"],
        normalized["graph_database_username"],
        normalized["graph_database_password"],
        normalized["graph_database_host"],
        normalized["graph_database_allow_anonymous"],
        normalized["graph_database_port"],
        normalized["graph_database_key"],
        normalized["graph_dataset_database_handler"],
        normalized["graph_database_subprocess_enabled"],
        normalized["kuzu_num_threads"],
        normalized["kuzu_buffer_pool_size"],
        normalized["kuzu_max_db_size"],
    )


def evict_graph_engines_for_database(graph_database_name: str) -> int:
    """Evict every cached graph engine bound to *graph_database_name*.

    The same per-dataset database can be cached under multiple keys: the
    dataset-handler creation key and the pipeline's context-config key differ
    in ``graph_file_path`` and ``graph_dataset_database_handler``, so key-exact
    ``evict_graph_engine`` misses the pipeline's entry and leaves an engine
    whose connection pool died with the dropped database. Per-dataset database
    names are dataset UUIDs, so matching the name against key fields cannot
    collide with other entries.

    Returns the number of evicted entries.
    """
    if not graph_database_name:
        raise ValueError("graph_database_name must be a non-empty database name")
    return _create_graph_engine.cache_evict_matching(graph_database_name=graph_database_name)


def is_graph_engine_cached(**kwargs) -> bool:
    """Check whether a graph engine entry exists in the cache without creating."""
    normalized = _normalize_optional_create_graph_engine_params(kwargs)
    provider = _normalize_graph_database_provider(kwargs.get("graph_database_provider"))
    return _create_graph_engine.cache_contains(
        provider,
        kwargs.get("graph_file_path"),
        normalized["graph_database_url"],
        normalized["graph_database_name"],
        normalized["graph_database_username"],
        normalized["graph_database_password"],
        normalized["graph_database_host"],
        normalized["graph_database_allow_anonymous"],
        normalized["graph_database_port"],
        normalized["graph_database_key"],
        normalized["graph_dataset_database_handler"],
        normalized["graph_database_subprocess_enabled"],
        normalized["kuzu_num_threads"],
        normalized["kuzu_buffer_pool_size"],
        normalized["kuzu_max_db_size"],
    )


@closing_lru_cache(maxsize=DATABASE_MAX_LRU_CACHE_SIZE)
def _create_graph_engine(
    graph_database_provider,
    graph_file_path,
    graph_database_url="",
    graph_database_name="",
    graph_database_username="",
    graph_database_password="",
    graph_database_host="",
    graph_database_allow_anonymous=False,
    graph_database_port="",
    graph_database_key="",
    graph_dataset_database_handler="",
    graph_database_subprocess_enabled=False,
    kuzu_num_threads=0,
    kuzu_buffer_pool_size=DEFAULT_KUZU_BUFFER_POOL_SIZE,
    kuzu_max_db_size=DEFAULT_KUZU_MAX_DB_SIZE,
):
    """
    Create a graph engine based on the specified provider type.

    This factory function initializes and returns the appropriate graph client depending on
    the database provider specified. It validates required parameters and raises an
    EnvironmentError if any are missing for the respective provider implementations.

    Parameters:
    -----------

        - graph_database_provider: The type of graph database provider to use (e.g., neo4j, falkor, ladybug).
        - graph_database_url: The URL for the graph database instance. Required for neo4j and falkordb providers.
        - graph_database_username: The username for authentication with the graph database.
          Required for neo4j provider.
        - graph_database_password: The password for authentication with the graph database.
          Required for neo4j provider.
        - graph_database_port: The port number for the graph database connection. Required
          for the falkordb provider
        - graph_file_path: The filesystem path to the graph file. Required for the ladybug
          provider.

    Returns:
    --------

        Returns an instance of the appropriate graph adapter depending on the provider type
        specified.
    """

    if graph_database_provider in supported_databases:
        adapter = supported_databases[graph_database_provider]

        return adapter(
            graph_database_url=graph_database_url,
            graph_database_username=graph_database_username,
            graph_database_password=graph_database_password,
            graph_database_port=graph_database_port,
            graph_database_key=graph_database_key,
            database_name=graph_database_name,
        )

    if graph_database_provider == "neo4j":
        if not graph_database_url:
            raise EnvironmentError("Missing required Neo4j URL.")

        from .neo4j_driver.adapter import Neo4jAdapter

        return Neo4jAdapter(
            graph_database_url=graph_database_url,
            graph_database_username=graph_database_username or None,
            graph_database_password=graph_database_password or None,
            graph_database_name=graph_database_name or None,
            graph_database_allow_anonymous=graph_database_allow_anonymous,
        )

    elif graph_database_provider == "postgres":
        from cognee.context_global_variables import backend_access_control_enabled

        if backend_access_control_enabled():
            if not (
                graph_database_host
                and graph_database_port
                and graph_database_username
                and graph_database_password
            ):
                raise EnvironmentError("Missing required Postgres graph credentials.")

            connection_string: str = (
                f"postgresql+asyncpg://{graph_database_username}:{graph_database_password}"
                f"@{graph_database_host}:{graph_database_port}/{graph_database_name}"
            )
        else:
            if (
                graph_database_port
                and graph_database_username
                and graph_database_password
                and graph_database_host
                and graph_database_name
            ):
                connection_string: str = (
                    f"postgresql+asyncpg://{graph_database_username}:{graph_database_password}"
                    f"@{graph_database_host}:{graph_database_port}/{graph_database_name}"
                )
            else:
                from cognee.infrastructure.databases.relational import get_relational_config

                logger.warning(
                    "Postgres graph credentials are not fully configured; "
                    "falling back to the relational database configuration. "
                    "Set GRAPH_DATABASE_HOST/PORT/USERNAME/PASSWORD/NAME explicitly "
                    "to avoid this fallback."
                )

                relational_config = get_relational_config()
                db_username = relational_config.db_username
                db_password = relational_config.db_password
                db_host = relational_config.db_host
                db_port = relational_config.db_port
                db_name = relational_config.db_name

                if not (db_host and db_port and db_name and db_username and db_password):
                    raise EnvironmentError("Missing required Postgres graph credentials!")

                connection_string: str = (
                    f"postgresql+asyncpg://{db_username}:{db_password}"
                    f"@{db_host}:{db_port}/{db_name}"
                )

        from .postgres.adapter import PostgresAdapter

        return PostgresAdapter(connection_string=connection_string)

    elif graph_database_provider in ("ladybug", "kuzu"):
        if not graph_file_path:
            raise EnvironmentError("Missing required Ladybug database path.")

        from .ladybug.adapter import LadybugAdapter

        if graph_database_subprocess_enabled:
            return LadybugAdapter.create_subprocess(
                db_path=graph_file_path,
                kuzu_num_threads=kuzu_num_threads,
                kuzu_buffer_pool_size=kuzu_buffer_pool_size,
                kuzu_max_db_size=kuzu_max_db_size,
            )

        return LadybugAdapter(
            db_path=graph_file_path,
            kuzu_num_threads=kuzu_num_threads,
            kuzu_buffer_pool_size=kuzu_buffer_pool_size,
            kuzu_max_db_size=kuzu_max_db_size,
        )

    elif graph_database_provider in ("ladybug-remote", "kuzu-remote"):
        if not graph_database_url:
            raise EnvironmentError("Missing required Ladybug remote URL.")

        from .ladybug.remote_ladybug_adapter import RemoteLadybugAdapter

        return RemoteLadybugAdapter(
            api_url=graph_database_url,
            username=graph_database_username,
            password=graph_database_password,
        )
    elif graph_database_provider == "neptune":
        try:
            from langchain_aws import NeptuneAnalyticsGraph
        except ImportError:
            raise ImportError(
                "langchain_aws is not installed. Please install it with 'pip install langchain_aws'"
            )

        if not graph_database_url:
            raise EnvironmentError("Missing Neptune endpoint.")

        from .neptune_driver.adapter import NeptuneGraphDB, NEPTUNE_ENDPOINT_URL

        if not graph_database_url.startswith(NEPTUNE_ENDPOINT_URL):
            raise ValueError(
                f"Neptune endpoint must have the format {NEPTUNE_ENDPOINT_URL}<GRAPH_ID>"
            )

        graph_identifier = graph_database_url.replace(NEPTUNE_ENDPOINT_URL, "")

        return NeptuneGraphDB(
            graph_id=graph_identifier,
        )

    elif graph_database_provider == "neptune_analytics":
        """
        Creates a graph DB from config
        We want to use a hybrid (graph & vector) DB and we should update this
        to make a single instance of the hybrid configuration (with embedder)
        instead of creating the hybrid object twice.
        """
        try:
            from langchain_aws import NeptuneAnalyticsGraph
        except ImportError:
            raise ImportError(
                "langchain_aws is not installed. Please install it with 'pip install langchain_aws'"
            )

        if not graph_database_url:
            raise EnvironmentError("Missing Neptune endpoint.")

        from ..hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
            NeptuneAnalyticsAdapter,
            NEPTUNE_ANALYTICS_ENDPOINT_URL,
        )

        if not graph_database_url.startswith(NEPTUNE_ANALYTICS_ENDPOINT_URL):
            raise ValueError(
                f"Neptune endpoint must have the format '{NEPTUNE_ANALYTICS_ENDPOINT_URL}<GRAPH_ID>'"
            )

        graph_identifier = graph_database_url.replace(NEPTUNE_ANALYTICS_ENDPOINT_URL, "")

        return NeptuneAnalyticsAdapter(
            graph_id=graph_identifier,
        )

    all_providers = list(supported_databases.keys()) + [
        "neo4j",
        "ladybug",
        "ladybug-remote",
        "kuzu",
        "kuzu-remote",
        "postgres",
        "neptune",
        "neptune_analytics",
    ]
    raise EnvironmentError(
        f"Unsupported graph database provider: {graph_database_provider}. "
        f"Supported providers are: {', '.join(all_providers)}"
    )
