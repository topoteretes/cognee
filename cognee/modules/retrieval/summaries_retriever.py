from typing import Any, Optional, List, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
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

    def __init__(self, top_k: int = 5, session_id: Optional[str] = None):
        """Initialize retriever with search parameters."""
        self.top_k = top_k
        self.session_id = session_id

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves text summary objects based on the query.

        On encountering a missing collection, raises NoDataError with a message to add data
        first.

        Parameters:
        -----------

            - query (str): The search query for which to retrieve summary context.

        Returns:
        --------

            - Any: A list of text summaries retrieved from the search.
        """
        logger.info(
            f"Starting summary retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        vector_engine = get_vector_engine()

        try:
            summaries_results = await vector_engine.search(
                "TextSummary_text", query, limit=self.top_k, include_payload=True
            )
            logger.info(f"Found {len(summaries_results)} summaries from vector search")

            await update_node_access_timestamps(summaries_results)

            return summaries_results
        except CollectionNotFoundError as error:
            logger.error("TextSummary_text collection not found in vector database")
            raise NoDataError("No data found in the system, please add data first.") from error

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves relevant summaries as context.

        Fetches text summaries based on a query from a vector engine and combines their text.
        Returns empty string if no summaries are found. Raises NoDataError if the collection is not
        found.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant text summaries.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved summaries, or an
              empty string if none are found.
        """
        if retrieved_objects:
            summary_payload_texts = [summary.payload["text"] for summary in retrieved_objects]
            return "\n".join(summary_payload_texts)
        else:
            return ""

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Generates a completion using text summaries.
        In case of the Summaries Retriever, we do not generate a completion, we just return
        the payloads of found summaries.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - retrieved_objects (Any): The retrieved objects to be used for generating a completion.
            - context (Any): The context to be used for generating a completion.

        Returns:
        --------

            - List[dict]: A list of payloads of found summaries.
        """
        # TODO: Do we want to generate a completion using LLM here?
        if retrieved_objects:
            summary_payloads = [summary.payload for summary in retrieved_objects]
            logger.info(f"Returning {len(summary_payloads)} summary payloads")
            return summary_payloads
        else:
            return []
