from abc import ABC, abstractmethod
from typing import Any, Optional, Callable


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

    @classmethod
    def as_search(cls) -> Callable:
        """Creates a search function from the retriever class."""
        return lambda query: cls().get_completion(query)
