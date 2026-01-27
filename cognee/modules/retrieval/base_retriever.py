from abc import ABC, abstractmethod
from typing import Any, Optional, Type, List


class BaseRetriever(ABC):
    """
    Base class for all retrieval operations.

    The retrieval workflow follows a three-step pipeline:
    1. get_retrieved_objects: Fetch raw data (e.g., Graph Edges, Vector chunks).
    2. get_context: Process raw data into a format suitable for an LLM (e.g., text string).
    3. get_completion: Generate a final response with the help of an LLM using the context and original query.
    """

    @abstractmethod
    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves the raw data points from the underlying storage (Graph or Vector DB).

        Args:
            query (str): The search query or input string.

        Returns:
            List[Any]: A list of raw objects (e.g., Edge objects, Document chunks)
                       relevant to the query.
        """
        pass

    @abstractmethod
    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Transforms raw retrieved objects into a structured context for the LLM.

        Args:
            query (str): The search query or input string.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.

        Returns:
            Any: The formatted context (typically a string or a list of strings)
                 to be injected into a prompt.
        """
        pass

    @abstractmethod
    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: Any,
        context: Any,
        session_id: Optional[str],
        response_model: Type = str,
    ) -> List[str]:
        """
        Generates a final output or answer based on the query and retrieved context.

        Args:
            query (str): The original user query.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.
            context (Optional[Any]): The formatted context string/data used to
                augment the generation. Output from get_context_from_objects.
            session_id (Optional[str]): Unique identifier for conversation history
                and session-based caching.
            response_model (Type): The expected return type or Pydantic model for
                structured outputs. Defaults to str.

        Returns:
            List[Any]: A list containing the generated completions or response objects.
        """
        pass
