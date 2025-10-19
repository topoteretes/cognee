"""Factory function to get the appropriate graph client based on the graph type."""

from functools import lru_cache

from .config import get_graph_context_config
from .graph_db_interface import GraphDBInterface
from .supported_databases import supported_databases


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


@lru_cache
def create_graph_engine(
    graph_database_provider,
    graph_file_path,
    graph_database_url="",
    graph_database_name="",
    graph_database_username="",
    graph_database_password="",
    graph_database_port="",
):
    """
    Create a graph engine based on the specified provider type.

    This factory function initializes and returns the appropriate graph client depending on
    the database provider specified. It validates required parameters and raises an
    EnvironmentError if any are missing for the respective provider implementations.

    Parameters:
    -----------

        - graph_database_provider: The type of graph database provider to use (e.g., neo4j, falkor, kuzu).
        - graph_database_url: The URL for the graph database instance. Required for neo4j and falkordb providers.
        - graph_database_username: The username for authentication with the graph database.
          Required for neo4j provider.
        - graph_database_password: The password for authentication with the graph database.
          Required for neo4j provider.
        - graph_database_port: The port number for the graph database connection. Required
          for the falkordb provider
        - graph_file_path: The filesystem path to the graph file. Required for the kuzu
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
        )

    elif graph_database_provider == "kuzu":
        if not graph_file_path:
            raise EnvironmentError("Missing required Kuzu database path.")

        from .kuzu.adapter import KuzuAdapter

        return KuzuAdapter(db_path=graph_file_path)

    elif graph_database_provider == "kuzu-remote":
        if not graph_database_url:
            raise EnvironmentError("Missing required Kuzu remote URL.")

        from .kuzu.remote_kuzu_adapter import RemoteKuzuAdapter

        return RemoteKuzuAdapter(
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

    raise EnvironmentError(
        f"Unsupported graph database provider: {graph_database_provider}. "
        f"Supported providers are: {', '.join(list(supported_databases.keys()) + ['neo4j', 'kuzu', 'kuzu-remote', 'neptune', 'neptune_analytics'])}"
    )
