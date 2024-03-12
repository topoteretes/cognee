


def search_categories(G, category):
    """
    Filter nodes by category.

    Parameters:
    - G (nx.Graph): The graph from which to filter nodes.
    - category (str): The category to filter nodes by.

    Returns:
    - list: A list of nodes that belong to the specified category.
    """
    return [node for node, data in G.nodes(data=True) if data.get('category') == category]
