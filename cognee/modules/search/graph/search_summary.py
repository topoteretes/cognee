


async def search_summary(graph, query:str, other_param:str = None):
    """
    Filter nodes that contain 'SUMMARY' in their identifiers and return their summary attributes.

    Parameters:
    - G (nx.Graph): The graph from which to filter nodes.

    Returns:
    - dict: A dictionary where keys are nodes containing 'SUMMARY' in their identifiers,
            and values are their 'summary' attributes.
    """
    return {node: data.get('summary') for node, data in graph.nodes(data=True) if 'SUMMARY' in node and 'summary' in data}



