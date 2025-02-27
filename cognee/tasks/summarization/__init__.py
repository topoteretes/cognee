from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from .summarize_code import summarize_code
from .summarize_text import summarize_text

# Instantiate retriever
summaries_retriever = SummariesRetriever()


# Define async function to expose retrieval functionality
async def query_summaries(query: str):
    return await summaries_retriever.get_completion(query)
