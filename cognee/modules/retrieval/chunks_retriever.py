from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError


class ChunksRetriever(BaseRetriever):
    """Retriever for handling document chunk-based searches."""

    def __init__(
        self,
        top_k: Optional[int] = 5,
    ):
        self.top_k = top_k

    async def get_context(self, query: str) -> Any:
        """Retrieves document chunks context based on the query."""
        vector_engine = get_vector_engine()

        try:
            found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=self.top_k)
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

        return [result.payload for result in found_chunks]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using document chunks context."""
        if context is None:
            context = await self.get_context(query)
        return context
