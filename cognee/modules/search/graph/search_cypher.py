
import networkx as nx
from typing import Union
from cognee.shared.data_models import GraphDBType
from cognee.infrastructure.databases.graph.config import get_graph_config

async def search_cypher(query:str, graph: Union[nx.Graph, any]):
    """
    Use a Cypher query to search the graph and return the results.
    """
    graph_config = get_graph_config()

    if graph_config.graph_engine == GraphDBType.NEO4J:
        result = await graph.run(query)
        return result

    else:
        raise ValueError("Unsupported graph engine type.")
