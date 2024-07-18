from typing import Union, Dict
import networkx as nx
from cognee.modules.search.llm.extraction.categorize_relevant_summary import categorize_relevant_summary
from cognee.shared.data_models import ChunkSummaries
from cognee.infrastructure.databases.vector import get_vector_engine

async def search_summary(query: str, graph: Union[nx.Graph, any]) -> Dict[str, str]:
    """
    Parameters:
    - graph (Union[nx.Graph, AsyncSession]): The graph object or Neo4j session.
    - query (str): The query string to filter nodes by, e.g., 'SUMMARY'.
    - other_param (str, optional): An additional parameter, unused in this implementation but could be for future enhancements.

    Returns:
    - Dict[str, str]: A dictionary where keys are node identifiers containing the query string, and values are their 'summary' attributes.
    """
    vector_engine = get_vector_engine()

    summaries_results = await vector_engine.search("chunk_summaries", query, limit = 5)
    summaries = [{
        "text": summary.payload["text"],
        "chunk_id": summary.payload["chunk_id"],
      } for summary in summaries_results]

    result = await categorize_relevant_summary(
        query = query,
        summaries = summaries,
        response_model = ChunkSummaries,
    )

    return result.summaries
