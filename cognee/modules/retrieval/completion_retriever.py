from typing import Any, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

logger = get_logger("CompletionRetriever")


class CompletionRetriever(BaseRetriever):
    """
    Retriever for handling LLM-based completion searches.

    Public methods:
    - get_context(query: str) -> str
    - get_completion(query: str, context: Optional[Any] = None) -> Any
    """

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        top_k: Optional[int] = 1,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 1

    async def get_context(self, query: str) -> str:
        """
        Retrieves relevant document chunks as context.

        Fetches document chunks based on a query from a vector engine and combines their text.
        Returns empty string if no chunks are found. Raises NoDataError if the collection is not
        found.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant document chunks.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved document chunks, or an
              empty string if none are found.
        """
        vector_engine = get_vector_engine()

        try:
            found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=self.top_k)

            if len(found_chunks) == 0:
                return ""

            # Combine all chunks text returned from vector search (number of chunks is determined by top_k
            chunks_payload = [found_chunk.payload["text"] for found_chunk in found_chunks]
            combined_context = "\n".join(chunks_payload)
            return combined_context
        except CollectionNotFoundError as error:
            logger.error("DocumentChunk_text collection not found")
            raise NoDataError("No data found in the system, please add data first.") from error

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Generates an LLM completion using the context.

        Retrieves context if not provided and generates a completion based on the query and
        context using an external completion generator.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - context (Optional[Any]): Optional pre-fetched context to use for generating the
              completion; if None, it retrieves the context for the query. (default None)

        Returns:
        --------

            - Any: The generated completion based on the provided query and context.
        """
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query, context, self.user_prompt_path, self.system_prompt_path
        )
        return [completion]
