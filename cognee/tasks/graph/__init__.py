from .extract_graph_from_data import extract_graph_from_data
from .extract_graph_from_code import extract_graph_from_code
from cognee.modules.retrieval.insights_retriever import InsightsRetriever

# Instantiate retriever
insights_retriever = InsightsRetriever()


# Define async function to expose retrieval functionality
async def query_graph_connections(query: str):
    """Replaces query_graph_connections with the new retriever approach."""
    return await insights_retriever.get_completion(query)
