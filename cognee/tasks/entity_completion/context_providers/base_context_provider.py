from abc import ABC, abstractmethod
from typing import List

from cognee.modules.engine.models import Entity


class BaseContextProvider(ABC):
    """Base class for context retrieval strategies."""

    @abstractmethod
    async def get_context(self, entities: List[Entity], query: str) -> str:
        """Get relevant context based on extracted entities and original query."""
        pass
