


from typing import Union, Dict
import networkx as nx
from cognee.shared.data_models import GraphDBType

async def search_summary(graph: Union[nx.Graph, any], query: str, infrastructure_config: Dict, other_param: str = None) -> Dict[str, str]:
    """
    Filter nodes based on a condition (such as containing 'SUMMARY' in their identifiers) and return their summary attributes.
    Supports both NetworkX graphs and Neo4j graph databases based on the configuration.

    Parameters:
    - graph (Union[nx.Graph, AsyncSession]): The graph object or Neo4j session.
    - query (str): The query string to filter nodes by, e.g., 'SUMMARY'.
    - infrastructure_config (Dict): Configuration that includes the graph engine type.
    - other_param (str, optional): An additional parameter, unused in this implementation but could be for future enhancements.

    Returns:
    - Dict[str, str]: A dictionary where keys are node identifiers containing the query string, and values are their 'summary' attributes.
    """
    if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
        return {node: data.get('summary') for node, data in graph.nodes(data=True) if query in node and 'summary' in data}

    elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
        cypher_query = f"""
        MATCH (n)
        WHERE n.id CONTAINS $query AND EXISTS(n.summary)
        RETURN n.id AS nodeId, n.summary AS summary
        """
        results = await graph.run(cypher_query, query=query)
        summary_data = {record["nodeId"]: record["summary"] for record in await results.list()}
        return summary_data

    else:
        raise ValueError("Unsupported graph engine type in the configuration.")
