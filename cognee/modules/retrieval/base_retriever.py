from abc import ABC, abstractmethod
from typing import Any, Optional, Type, List, Union


class BaseRetriever(ABC):
    """
    Base class for all retrieval operations.

    The retrieval workflow follows a three-step pipeline:
    1. get_retrieved_objects: Fetch raw data (e.g., Graph Edges, Vector chunks).
    2. get_context: Process raw data into a format suitable for an LLM (e.g., text string).
    3. get_completion: Generate a final response with the help of an LLM using the context and original query.
    """

    @abstractmethod
    async def get_retrieved_objects(self, query: Optional[str], query_batch: Optional[str]) -> Any:
        """
        Retrieves the raw data points from the underlying storage (Graph or Vector DB).

        Args:
            query (str): The search query or input string.
            query_batch (List[str]): The batch of search queries.

        Returns:
            List[Any]: A list of raw objects (e.g., Edge objects, Document chunks)
                       relevant to the query.
        """
        pass

    @abstractmethod
    async def get_context_from_objects(
        self, retrieved_objects: Any, query: Optional[str] = None, query_batch: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Transforms raw retrieved objects into a structured context for the LLM.

        Args:
            query (str): The search query or input string.
            query_batch (List[str]): The batch of search queries.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.

        Returns:
            Any: The formatted context (typically a string or a list of strings)
                 to be injected into a prompt.
        """
        pass

    @abstractmethod
    async def get_completion_from_context(
        self,
        retrieved_objects: Any,
        context: Any,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
    ) -> Union[List[str], List[dict]]:
        """
        Generates a final output or answer based on the query and retrieved context.

        Args:
            query (str): The original user query.
            query_batch (List[str]): The batch of original user queries.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.
            context (Optional[Any]): The formatted context string/data used to
                augment the generation. Output from get_context_from_objects.

        Returns:
            List[Any]: A list containing the generated completions or response objects.
        """
        pass
