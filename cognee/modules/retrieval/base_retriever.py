from abc import ABC, abstractmethod
from typing import Any, Optional, Type, List


class BaseRetriever(ABC):
    """Base class for all retrieval operations."""

    @abstractmethod
    async def get_context(self, query: str) -> Any:
        """Retrieves context based on the query."""
        pass

    @abstractmethod
    async def get_completion(
        self,
        query: str,
        context: Optional[Any] = None,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ) -> List[Any]:
        """Generates a response using the query and optional context."""
        pass
