from typing import Union, Dict
import networkx as nx
from cognee.infrastructure.databases.graph import get_graph_engine

async def search_adjacent(graph: Union[nx.Graph, any], query: str, other_param: dict = None) -> Dict[str, str]:
    """
    Find the neighbours of a given node in the graph and return their ids and descriptions.

    Parameters:
    - graph (Union[nx.Graph, AsyncSession]): The graph object or Neo4j session.
    - query (str): Unused in this implementation but could be used for future enhancements.
    - other_param (dict, optional): A dictionary that may contain 'node_id' to specify the node.

    Returns:
    - Dict[str, str]: A dictionary containing the unique identifiers and descriptions of the neighbours of the given node.
    """
    node_id = other_param.get('node_id') if other_param else query

    if node_id is None:
        return {}

    graph_engine = await get_graph_engine()

    return await graph_engine.get_neighbours(node_id)
