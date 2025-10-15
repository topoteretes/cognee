from typing import Any, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError

logger = get_logger("SummariesRetriever")


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
        logger.info(
            f"Starting summary retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        vector_engine = get_vector_engine()

        try:
            summaries_results = await vector_engine.search(
                "TextSummary_text", query, limit=self.top_k
            )
            logger.info(f"Found {len(summaries_results)} summaries from vector search")
        except CollectionNotFoundError as error:
            logger.error("TextSummary_text collection not found in vector database")
            raise NoDataError("No data found in the system, please add data first.") from error

        summary_payloads = [summary.payload for summary in summaries_results]
        logger.info(f"Returning {len(summary_payloads)} summary payloads")
        return summary_payloads

    async def get_completion(
        self, query: str, context: Optional[Any] = None, session_id: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Generates a completion using summaries context.

        If no context is provided, retrieves context using the query. Returns the provided
        context or the retrieved context if none was given.

        Parameters:
        -----------

            - query (str): The search query for generating the completion.
            - context (Optional[Any]): Optional context for the completion; if not provided,
              will be retrieved based on the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: The generated completion context, which is either provided or retrieved.
        """
        logger.info(
            f"Starting completion generation for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        if context is None:
            logger.debug("No context provided, retrieving context from vector database")
            context = await self.get_context(query)
        else:
            logger.debug("Using provided context")

        logger.info(
            f"Returning context with {len(context) if isinstance(context, list) else 1} item(s)"
        )
        return context
