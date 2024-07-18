from typing import Union, Dict
import networkx as nx
from cognee.infrastructure.databases.vector import get_vector_engine

async def search_similarity(query: str, graph: Union[nx.Graph, any]) -> Dict[str, str]:
    """
    Parameters:
    - graph (Union[nx.Graph, AsyncSession]): The graph object or Neo4j session.
    - query (str): The query string to filter nodes by, e.g., 'SUMMARY'.

    Returns:
    - Dict[str, str]: A dictionary where keys are node identifiers containing the query string, and values are their 'result' attributes.
    """
    vector_engine = get_vector_engine()

    similar_results = await vector_engine.search("chunks", query, limit = 5)
    results = [{
        "text": result.payload["text"],
        "chunk_id": result.payload["chunk_id"],
      } for result in similar_results]

    return results
