from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


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
            return "\n".join(chunks_payload)
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Generates an LLM completion using the context.

        Retrieves context if not provided and generates a completion based on the query and
        context using an external completion generator.

        Parameters:
        -----------

            - query (str): The input query for which the completion is generated.
            - context (Optional[Any]): Optional context to use for generating the completion; if
              not provided, it will be retrieved using get_context. (default None)

        Returns:
        --------

            - Any: A list containing the generated completion from the LLM.
        """
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]
