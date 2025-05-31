from abc import ABC, abstractmethod
from typing import List

from cognee.infrastructure.engine import DataPoint


class BaseContextProvider(ABC):
    """
    Base class for context retrieval strategies.

    This class serves as a blueprint for creating different strategies that retrieve context
    based on specified entities and a query. Any subclass must implement the `get_context`
    method to define how context is retrieved.
    """

    @abstractmethod
    async def get_context(self, entities: List[DataPoint], query: str) -> str:
        """
        Get relevant context based on extracted entities and original query.

        This method must be implemented by subclasses to define the logic for retrieving context
        relevant to the provided `entities` and `query`. It is an asynchronous method that will
        return a string representing the context.

        Parameters:
        -----------

            - entities (List[DataPoint]): A list of data points representing extracted entities
              relevant to the query.
            - query (str): The original query string used to extract the entities and retrieve
              context.
        """
        pass
