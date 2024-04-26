"""Factory function to get the appropriate graph client based on the graph type."""

from cognee.config import Config
from cognee.shared.data_models import GraphDBType
from cognee.infrastructure import infrastructure_config
from .graph_db_interface import GraphDBInterface
from .networkx.adapter import NetworkXAdapter

config = Config()
config.load()


async def get_graph_client(graph_type: GraphDBType, graph_file_name: str = None) -> GraphDBInterface :
    """Factory function to get the appropriate graph client based on the graph type."""
    graph_file_path = f"{infrastructure_config.get_config('database_directory_path')}/{graph_file_name if graph_file_name else config.graph_filename}"

    if graph_type == GraphDBType.NEO4J:
        try:
            from .neo4j_driver.adapter import Neo4jAdapter

            return Neo4jAdapter(
                graph_database_url = config.graph_database_url,
                graph_database_username = config.graph_database_username,
                graph_database_password = config.graph_database_password
            )
        except:
            pass
            
    graph_client = NetworkXAdapter(filename = graph_file_path)


    if (graph_client.graph is None):
        await graph_client.load_graph_from_file()

    return graph_client
