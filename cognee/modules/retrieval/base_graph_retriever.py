from typing import Any, List, Optional, Type
from abc import ABC, abstractmethod

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


class BaseGraphRetriever(ABC):
    """Base class for all graph based retrievers."""

    @abstractmethod
    async def get_context(self, query: str) -> List[Edge]:
        """Retrieves triplets based on the query."""
        pass

    @abstractmethod
    async def get_completion(
        self,
        query: str,
        context: Optional[List[Edge]] = None,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ) -> List[Any]:
        """Generates a response using the query and optional context (triplets)."""
        pass
