from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.tasks.completion.exceptions import NoRelevantDataFound

# Instantiate retrievers
completion_retriever = CompletionRetriever()
graph_completion_retriever = GraphCompletionRetriever()
graph_summary_completion_retriever = GraphSummaryCompletionRetriever()


# Define async functions to expose retrieval functionality
async def query_completion(query: str):
    return await completion_retriever.get_completion(query)


async def graph_query_completion(query: str):
    return await graph_completion_retriever.get_completion(query)


async def graph_query_summary_completion(query: str):
    return await graph_summary_completion_retriever.get_completion(query)


async def resolve_graph_context(triplets):
    return await graph_completion_retriever.resolve_edges_to_text(triplets)
