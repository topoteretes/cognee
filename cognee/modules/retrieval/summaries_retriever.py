from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever


class SummariesRetriever(BaseRetriever):
    """Retriever for handling summary-based searches."""

    def __init__(self, limit: int = 5):
        """Initialize retriever with search parameters."""
        self.limit = limit

    async def get_context(self, query: str) -> Any:
        """Retrieves summary context based on the query."""
        vector_engine = get_vector_engine()
        summaries_results = await vector_engine.search("TextSummary_text", query, limit=self.limit)
        return [summary.payload for summary in summaries_results]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using summaries context."""
        if context is None:
            context = await self.get_context(query)
        return context
