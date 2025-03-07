from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Dict, List


class BaseRetriever(ABC):
    """Base class for all retrieval operations."""

    @abstractmethod
    async def get_context(self, query: str) -> Any:
        """Retrieves context based on the query."""
        pass

    @abstractmethod
    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a response using the query and optional context."""
        pass
        
    async def search_vector_db(
        self, 
        query: str, 
        collection_name: str,
        limit: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the vector database for relevant documents.
        
        Args:
            query: The search query
            collection_name: The name of the collection to search
            limit: Maximum number of results to return
            filter_condition: Optional filter to apply to the search
            
        Returns:
            List of search results
        """
        from cognee.infrastructure.databases.vector import get_vector_engine
        vector_engine = get_vector_engine()
        results = await vector_engine.search(
            collection_name=collection_name,
            query_text=query,
            limit=limit,
            filter_condition=filter_condition
        )
        return results
