""" This module contains the function to find the neighbours of a given node in the graph"""


def search_adjacent(G, node_id:str)->dict:
    """ Find the neighbours of a given node in the graph
    :param G: A NetworkX graph object
    :param node_id: The unique identifier of the node
    :return: A dictionary containing the unique identifiers and descriptions of the neighbours of the given node
    """

    neighbors = list(G.neighbors(node_id))
    neighbor_descriptions = {}

    for neighbor in neighbors:
        # Access the 'description' attribute for each neighbor
        # The get method returns None if 'description' attribute does not exist for the node
        neighbor_descriptions[neighbor] = G.nodes[neighbor].get("description")

    return neighbor_descriptions