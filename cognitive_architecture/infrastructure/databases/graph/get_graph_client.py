"""Factory function to get the appropriate graph client based on the graph type."""
from typing import Type
from cognitive_architecture.config import Config
from .graph_db_interface import GraphDBInterface
from .networkx.adapter import NetworXAdapter
# Assuming Neo4jAdapter is defined somewhere
# from .neo4j.adapter import Neo4jAdapter
from enum import Enum, auto


config = Config()
config.load()

class GraphDBType(Enum):
    NETWORKX = auto()
    NEO4J = auto()
def get_graph_client(graph_type: GraphDBType, graph_filename: str) -> Type[GraphDBInterface]:
    """Factory function to get the appropriate graph client based on the graph type."""
    if graph_filename is not None:
        config.graph_filename = graph_filename
    if graph_type == GraphDBType.NETWORKX:
        return NetworXAdapter(filename = config.graph_filename)  # Adjust as needed for NetworkX adapter configuration
    elif graph_type == GraphDBType.NEO4J:
        # return Neo4jAdapter(config.neo4j_config)  # Uncomment and adjust as needed for Neo4j adapter configuration
        raise NotImplementedError("Neo4j adapter is not implemented yet.")
    else:
        raise ValueError("Unsupported graph database type.")