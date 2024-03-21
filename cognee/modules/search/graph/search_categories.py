


async def search_categories(graph, query:str, other_param:str = None):
    """
    Filter nodes that contain 'LABEL' in their identifiers and return their summary attributes.

    Parameters:
    - G (nx.Graph): The graph from which to filter nodes.

    Returns:
    - dict: A dictionary where keys are nodes containing 'SUMMARY' in their identifiers,
            and values are their 'summary' attributes.
    """
    return {node: data.get('content_labels') for node, data in graph.nodes(data=True) if 'LABEL' in node and 'content_labels' in data}



