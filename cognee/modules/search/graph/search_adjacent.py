""" This module contains the function to find the neighbours of a given node in the graph"""


async def search_adjacent(graph, query: str, other_param: dict = None) -> dict:
    """ Find the neighbours of a given node in the graph
    :param graph: A NetworkX graph object

    :return: A dictionary containing the unique identifiers and descriptions of the neighbours of the given node
    """

    node_id = other_param.get('node_id') if other_param else None

    if node_id is None or node_id not in graph:
        return {}

    neighbors = list(graph.neighbors(node_id))
    neighbor_descriptions = {}

    for neighbor in neighbors:
        # Access the 'description' attribute for each neighbor
        # The get method returns None if 'description' attribute does not exist for the node
        neighbor_descriptions[neighbor] = graph.nodes[neighbor].get('description')

    return neighbor_descriptions