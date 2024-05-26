from typing import Union, Dict
import re

from pydantic import BaseModel

from cognee.modules.search.llm.extraction.categorize_relevant_category import categorize_relevant_category

""" Search categories in the graph and return their summary attributes. """

from cognee.shared.data_models import GraphDBType, DefaultContentPrediction
import networkx as nx

from cognee.infrastructure.databases.graph.config import get_graph_config
graph_config = get_graph_config()
from cognee.infrastructure.databases.vector.config import get_vectordb_config
vector_config = get_vectordb_config()

def strip_exact_regex(s, substring):
    # Escaping substring to be used in a regex pattern
    pattern = re.escape(substring)
    # Regex to match the exact substring at the start and end
    return re.sub(f"^{pattern}|{pattern}$", "", s)


class DefaultResponseModel(BaseModel):
    document_id: str

async def search_categories(query:str, graph: Union[nx.Graph, any], query_label: str=None, infrastructure_config: Dict=None):
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
    from cognee.infrastructure import infrastructure_config
    if graph_config.graph_engine == GraphDBType.NETWORKX:

        categories_and_ids = [
            {'document_id': strip_exact_regex(_, "DATA_SUMMARY__"), 'Summary': data['summary']}
            for _, data in graph.nodes(data=True)
            if 'summary' in data
        ]
        connected_nodes = []
        for id in categories_and_ids:
            print("id", id)
            connected_nodes.append(list(graph.neighbors(id['document_id'])))
        check_relevant_category = await categorize_relevant_category(query, categories_and_ids, response_model=DefaultResponseModel )
        connected_nodes = list(graph.neighbors(check_relevant_category['document_id']))
        descriptions = {node: graph.nodes[node].get('description', 'No desc available') for node in connected_nodes}
        return descriptions

    elif graph_config.graph_engine == GraphDBType.NEO4J:
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
