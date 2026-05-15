"""Factory function to get the appropriate graph client based on the graph type."""

import inspect
import os
from numbers import Number

from cognee.infrastructure.databases.utils.closing_lru_cache import closing_lru_cache
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE

from .kuzu.adapter import DEFAULT_KUZU_BUFFER_POOL_SIZE, DEFAULT_KUZU_MAX_DB_SIZE
from .config import get_graph_context_config
from .graph_db_interface import GraphDBInterface
from .supported_databases import supported_databases


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
    """Factory function to get the appropriate graph client based on the graph type."""
    # Get appropriate graph configuration based on current async context
    config = get_graph_context_config()

    graph_client = create_graph_engine(**config)

    # Async functions can't be cached. After creating and caching the graph engine
    # handle all necessary async operations for different graph types bellow.

    # Run any adapter‐specific async initialization
    if hasattr(graph_client, "initialize"):
        await graph_client.initialize()

    return graph_client


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

    normalized_optional_params = _normalize_optional_create_graph_engine_params(locals())
    graph_database_url = normalized_optional_params["graph_database_url"]
    graph_database_provider = _normalize_graph_database_provider(graph_database_provider)
    graph_database_name = normalized_optional_params["graph_database_name"]
    graph_database_username = normalized_optional_params["graph_database_username"]
    graph_database_password = normalized_optional_params["graph_database_password"]
    graph_database_host = normalized_optional_params["graph_database_host"]
    graph_database_allow_anonymous = normalized_optional_params["graph_database_allow_anonymous"]
    graph_database_port = normalized_optional_params["graph_database_port"]
    graph_database_key = normalized_optional_params["graph_database_key"]
    graph_dataset_database_handler = normalized_optional_params["graph_dataset_database_handler"]
    # The Kuzu/subprocess params also went through ``_normalize_optional_*``;
    # reassign so callers passing ``None`` see the function-default applied
    # (otherwise ``None`` would flow into the cache key and the factory).
    graph_database_subprocess_enabled = normalized_optional_params[
        "graph_database_subprocess_enabled"
    ]
    kuzu_num_threads = normalized_optional_params["kuzu_num_threads"]
    kuzu_buffer_pool_size = normalized_optional_params["kuzu_buffer_pool_size"]
    kuzu_max_db_size = normalized_optional_params["kuzu_max_db_size"]

    # Check USE_UNIFIED_PROVIDER outside the cache so it's always re-read
    unified_provider = os.environ.get("USE_UNIFIED_PROVIDER", "")
    if unified_provider == "pghybrid":
        from .postgres.adapter import PostgresAdapter
        from cognee.infrastructure.databases.relational.get_relational_engine import (
            get_relational_engine,
        )

        return PostgresAdapter(connection_string=get_relational_engine().db_uri)

    return _create_graph_engine(
        graph_database_provider,
        graph_file_path,
        graph_database_url,
        graph_database_name,
        graph_database_username,
        graph_database_password,
        graph_database_host,
        graph_database_allow_anonymous,
        graph_database_port,
        graph_database_key,
        graph_dataset_database_handler,
        graph_database_subprocess_enabled,
        kuzu_num_threads,
        kuzu_buffer_pool_size,
        kuzu_max_db_size,
    )


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
