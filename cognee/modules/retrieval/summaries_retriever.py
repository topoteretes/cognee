from typing import Any, Optional, Dict, List

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever


class SummariesRetriever(BaseRetriever):
    """Retriever for handling summary-based searches."""

    def __init__(self, limit: int = 5):
        """Initialize retriever with search parameters."""
        self.limit = limit

    async def get_context(self, query: str) -> Any:
        """Retrieves summary context based on the query."""
        results = await self.search_vector_db(
            query,
            collection_name="summaries",
            limit=self.limit,
            filter_condition=None
        )
        
        # Transform the results to have a content key
        transformed_results = []
        for result in results:
            payload = result.get("payload", {})
            transformed_result = {
                "score": result.get("score", 0)
            }
            
            # Only add content if text exists in the payload
            if "text" in payload:
                transformed_result["content"] = payload["text"]
                
            # Only add document_id if it exists in the payload
            if "document_id" in payload:
                transformed_result["document_id"] = payload["document_id"]
                
            # Only add metadata if it exists in the payload
            if "metadata" in payload:
                transformed_result["metadata"] = payload["metadata"]
                
            transformed_results.append(transformed_result)
            
        return transformed_results

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using summaries context."""
        if context is None:
            context = await self.get_context(query)
        return context
