from typing import Union, Dict

""" Search categories in the graph and return their summary attributes. """

from cognee.shared.data_models import GraphDBType
import networkx as nx

async def search_categories(graph: Union[nx.Graph, any], query_label: str, infrastructure_config: Dict):
    """
    Filter nodes in the graph that contain the specified label and return their summary attributes.
    This function supports both NetworkX graphs and Neo4j graph databases.

    Parameters:
    - graph (Union[nx.Graph, AsyncSession]): The graph object or Neo4j session.
    - query_label (str): The label to filter nodes by.
    - infrastructure_config (Dict): Configuration that includes the graph engine type.

    Returns:
    - Union[Dict, List[Dict]]: For NetworkX, returns a dictionary where keys are node identifiers,
      and values are their 'content_labels' attributes. For Neo4j, returns a list of dictionaries,
      each representing a node with 'nodeId' and 'summary'.
    """
    # Determine which client is in use based on the configuration
    if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
        # Logic for NetworkX
        return {node: data.get('content_labels') for node, data in graph.nodes(data=True) if query_label in node and 'content_labels' in data}

    elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
        # Logic for Neo4j
        cypher_query = """
        MATCH (n)
        WHERE $label IN labels(n) AND EXISTS(n.summary)
        RETURN id(n) AS nodeId, n.summary AS summary
        """
        result = await graph.run(cypher_query, label=query_label)
        nodes_summary = [{"nodeId": record["nodeId"], "summary": record["summary"]} for record in await result.list()]
        return nodes_summary

    else:
        raise ValueError("Unsupported graph engine type.")
