


from typing import Union, Dict
import networkx as nx
from cognee.infrastructure import infrastructure_config

from cognee.modules.search.llm.extraction.categorize_relevant_summary import categorize_relevant_summary
from cognee.shared.data_models import GraphDBType, ResponseSummaryModel
from cognee.infrastructure.databases.graph.config import get_graph_config
graph_config = get_graph_config()
from cognee.infrastructure.databases.vector.config import get_vectordb_config
vector_config = get_vectordb_config()
import re

def strip_exact_regex(s, substring):
    # Escaping substring to be used in a regex pattern
    pattern = re.escape(substring)
    # Regex to match the exact substring at the start and end
    return re.sub(f"^{pattern}|{pattern}$", "", s)
async def search_summary( query: str,  graph: Union[nx.Graph, any]) -> Dict[str, str]:
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

    if graph_config.graph_engine == GraphDBType.NETWORKX:
        print("graph", graph)
        summaries_and_ids = [
            {'document_id': strip_exact_regex(_, "DATA_SUMMARY__"), 'Summary': data['summary']}
            for _, data in graph.nodes(data=True)
            if 'summary' in data
        ]
        print("summaries_and_ids", summaries_and_ids)
        check_relevant_summary = await categorize_relevant_summary(query, summaries_and_ids, response_model=ResponseSummaryModel)
        print("check_relevant_summary", check_relevant_summary)

        connected_nodes = list(graph.neighbors(check_relevant_summary['document_id']))
        print("connected_nodes", connected_nodes)
        descriptions = {node: graph.nodes[node].get('description', 'No desc available') for node in connected_nodes}
        print("descs", descriptions)
        return descriptions


    elif graph_config.graph_engine == GraphDBType.NEO4J:
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
