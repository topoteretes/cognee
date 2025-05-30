from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError


class SummariesRetriever(BaseRetriever):
    """
    Retriever for handling summary-based searches.

    Public methods:
    - __init__
    - get_context
    - get_completion

    Instance variables:
    - top_k: int - Number of top summaries to retrieve.
    """

    def __init__(self, top_k: int = 5):
        """Initialize retriever with search parameters."""
        self.top_k = top_k

    async def get_context(self, query: str) -> Any:
        """
        Retrieves summary context based on the query.

        On encountering a missing collection, raises NoDataError with a message to add data
        first.

        Parameters:
        -----------

            - query (str): The search query for which to retrieve summary context.

        Returns:
        --------

            - Any: A list of payloads from the retrieved summaries.
        """
        vector_engine = get_vector_engine()

        try:
            summaries_results = await vector_engine.search(
                "TextSummary_text", query, limit=self.top_k
            )
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

        return [summary.payload for summary in summaries_results]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Generates a completion using summaries context.

        If no context is provided, retrieves context using the query. Returns the provided
        context or the retrieved context if none was given.

        Parameters:
        -----------

            - query (str): The search query for generating the completion.
            - context (Optional[Any]): Optional context for the completion; if not provided,
              will be retrieved based on the query. (default None)

        Returns:
        --------

            - Any: The generated completion context, which is either provided or retrieved.
        """
        if context is None:
            context = await self.get_context(query)
        return context
