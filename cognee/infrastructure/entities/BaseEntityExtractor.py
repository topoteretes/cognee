from abc import ABC, abstractmethod
from typing import List

from cognee.modules.engine.models import Entity


class BaseEntityExtractor(ABC):
    """
    Base class for entity extraction strategies.

    This class defines the interface for entity extraction methods that derived classes must
    implement. It serves as a blueprint for entity extraction strategies.
    """

    @abstractmethod
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from the given text.

        This is an abstract method that must be implemented by any subclass of
        BaseEntityExtractor. The implementation should take a string as input and return a list
        of extracted entities, defined by the Entity type.

        Parameters:
        -----------

            - text (str): A string containing the text from which to extract entities.
        """
        pass
