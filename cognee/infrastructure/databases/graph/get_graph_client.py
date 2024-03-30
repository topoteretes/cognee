"""Factory function to get the appropriate graph client based on the graph type."""
from cognee.config import Config
from cognee.shared.data_models import GraphDBType
from cognee.infrastructure import infrastructure_config
from .graph_db_interface import GraphDBInterface
from .networkx.adapter import NetworXAdapter
# from .neo4j.adapter import Neo4jAdapter

config = Config()
config.load()


def get_graph_client(graph_type: GraphDBType, graph_filename: str=None) -> GraphDBInterface :
    """Factory function to get the appropriate graph client based on the graph type."""
    if graph_filename is None:
        graph_filename = f"{infrastructure_config.get_config()['database_directory_path']}/{config.graph_filename}"
    if graph_type == GraphDBType.NETWORKX:
        return  NetworXAdapter(filename = graph_filename)
    elif graph_type == GraphDBType.NEO4J:
        # return Neo4jAdapter(config.neo4j_config)  # Uncomment and adjust as needed for Neo4j adapter configuration
        raise NotImplementedError("Neo4j adapter is not implemented yet.")
    else:
        raise ValueError("Unsupported graph database type.")