"""Factory function to get the appropriate graph client based on the graph type."""
from functools import lru_cache

from .config import get_graph_config
from .graph_db_interface import GraphDBInterface


async def get_graph_engine() -> GraphDBInterface:
    """Factory function to get the appropriate graph client based on the graph type."""
    graph_client = create_graph_engine()

    # Async functions can't be cached. After creating and caching the graph engine
    # handle all necessary async operations for different graph types bellow.
    config = get_graph_config()

    # Handle loading of graph for NetworkX
    if config.graph_database_provider.lower() == "networkx" and graph_client.graph is None:
        await graph_client.load_graph_from_file()

    return graph_client


@lru_cache
def create_graph_engine() -> GraphDBInterface:
    """Factory function to create the appropriate graph client based on the graph type."""
    config = get_graph_config()

    if config.graph_database_provider == "neo4j":
        if not (config.graph_database_url and config.graph_database_username and config.graph_database_password):
            raise EnvironmentError("Missing required Neo4j credentials.")

        from .neo4j_driver.adapter import Neo4jAdapter

        return Neo4jAdapter(
            graph_database_url=config.graph_database_url,
            graph_database_username=config.graph_database_username,
            graph_database_password=config.graph_database_password
        )

    elif config.graph_database_provider == "falkordb":
        if not (config.graph_database_url and config.graph_database_port):
            raise EnvironmentError("Missing required FalkorDB credentials.")

        from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
        from cognee.infrastructure.databases.hybrid.falkordb.FalkorDBAdapter import FalkorDBAdapter

        embedding_engine = get_embedding_engine()

        return FalkorDBAdapter(
            database_url=config.graph_database_url,
            database_port=config.graph_database_port,
            embedding_engine=embedding_engine,
        )

    from .networkx.adapter import NetworkXAdapter
    graph_client = NetworkXAdapter(filename=config.graph_file_path)

    return graph_client
