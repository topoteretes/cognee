"""Factory function to get the appropriate graph client based on the graph type."""

import inspect
import os
import weakref
from numbers import Number

from cognee.infrastructure.databases.utils.closing_lru_cache import closing_lru_cache
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE
from cognee.shared.logging_utils import get_logger

from .kuzu.adapter import DEFAULT_KUZU_BUFFER_POOL_SIZE, DEFAULT_KUZU_MAX_DB_SIZE
from .config import get_graph_context_config
from .graph_db_interface import GraphDBInterface
from .supported_databases import supported_databases

logger = get_logger("GraphEngine")

# Engines whose idempotent ``initialize()`` (schema/constraint setup for
# Postgres/Neo4j) has already run this process. Keyed by the leased engine proxy
# (stable per cache entry) via a WeakSet so entries drop automatically when the
# engine is evicted + collected, and a freshly created engine for the same key
# re-initializes. Replaces the old ``_GraphEngineHandle._last_initialized_id``
# guard so we don't issue a redundant ``initialize()`` round-trip on every
# ``get_graph_engine()`` resolution. (Ladybug has no ``initialize`` — no effect.)
_INITIALIZED_ENGINES: "weakref.WeakSet" = weakref.WeakSet()


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


async def get_graph_engine() -> GraphDBInterface:
    """Resolve the graph engine for the current context and return the live
    adapter (a leased proxy from ``closing_lru_cache``).

    Resolution is asynchronous and goes through :func:`acreate_graph_engine` so
    that engine *creation* can ``await`` an in-flight close of the same cache
    key before constructing — see that function for why the factory is async.
    This is what makes the fix for the subprocess DB file-lock race (#3708)
    deterministic: a re-created engine never opens a DB path whose previous
    worker is still shutting down and holding the lock.

    Note: the returned adapter is a live reference for the *current* async
    scope. Callers should not stash it on a long-lived object and reuse it
    across a cache-invalidating operation (prune / delete / per-dataset context
    exit); call ``get_graph_engine()`` again after such an operation.
    """
    config = get_graph_context_config()
    engine = await acreate_graph_engine(**config)
    # Run the idempotent schema/constraint setup once per engine instance, not on
    # every resolve — guarded by ``_INITIALIZED_ENGINES`` (membership is by proxy
    # identity, so a re-created engine after eviction initializes again).
    if hasattr(engine, "initialize") and engine not in _INITIALIZED_ENGINES:
        await engine.initialize()
        _INITIALIZED_ENGINES.add(engine)
    return engine


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
    """Normalize engine parameters and return the positional argument tuple
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
    """Async counterpart of :func:`create_graph_engine`, used as the primary
    creation path (via :func:`get_graph_engine`).

    Why the factory is async
    ------------------------
    The subprocess-backed graph engine (Ladybug/Kuzu) holds an exclusive on-disk
    file lock for the worker's lifetime. Engines are cached in
    ``closing_lru_cache``; when an entry is evicted its ``close()`` (worker
    shutdown, which releases the lock) runs asynchronously. A *synchronous*
    factory has no way to wait for that close, so re-creating an engine for the
    same DB path could spawn a second worker that races the still-closing first
    one and fails with "Could not set lock on file" (#3708).

    Making creation ``async`` lets the cache's ``aget_or_create`` **await the
    in-flight close of the same key on the caller's event loop** before
    constructing — the awaiting creator yields the loop so the close task runs
    to completion (worker exits, lock released) first. This is what removes the
    race deterministically, without needing an off-loop close thread or a
    re-resolving engine handle.

    Delegates to :func:`create_graph_engine`'s normalization
    (:func:`_resolve_graph_engine_args`) so the cache key is identical to the
    sync path.
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
