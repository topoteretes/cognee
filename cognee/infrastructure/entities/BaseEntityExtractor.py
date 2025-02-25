from abc import ABC, abstractmethod
from typing import List

from cognee.modules.engine.models import Entity


class BaseEntityExtractor(ABC):
    """Base class for entity extraction strategies."""

    @abstractmethod
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from the given text."""
        pass
