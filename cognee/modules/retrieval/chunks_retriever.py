from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever


class ChunksRetriever(BaseRetriever):
    """Retriever for handling document chunk-based searches."""

    async def get_context(self, query: str) -> Any:
        """Retrieves document chunks context based on the query."""
        vector_engine = get_vector_engine()
        found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=5)
        return [result.payload for result in found_chunks]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using document chunks context."""
        if context is None:
            context = await self.get_context(query)
        return context
