from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError


class SummariesRetriever(BaseRetriever):
    """Retriever for handling summary-based searches."""

    def __init__(self, top_k: int = 5):
        """Initialize retriever with search parameters."""
        self.top_k = top_k

    async def get_context(self, query: str) -> Any:
        """Retrieves summary context based on the query."""
        vector_engine = get_vector_engine()

        try:
            summaries_results = await vector_engine.search(
                "TextSummary_text", query, limit=self.top_k
            )
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

        return [summary.payload for summary in summaries_results]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using summaries context."""
        if context is None:
            context = await self.get_context(query)
        return context
