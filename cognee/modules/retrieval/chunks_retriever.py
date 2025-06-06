from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError


class ChunksRetriever(BaseRetriever):
    """
    Handles document chunk-based searches by retrieving relevant chunks and generating
    completions from them.

    Public methods:

    - get_context: Retrieves document chunks based on a query.
    - get_completion: Generates a completion using provided context or retrieves context if
    not given.
    """

    def __init__(
        self,
        top_k: Optional[int] = 5,
    ):
        self.top_k = top_k

    async def get_context(self, query: str) -> Any:
        """
        Retrieves document chunks context based on the query.

        Searches for document chunks relevant to the specified query using a vector engine.
        Raises a NoDataError if no data is found in the system.

        Parameters:
        -----------

            - query (str): The query string to search for relevant document chunks.

        Returns:
        --------

            - Any: A list of document chunk payloads retrieved from the search.
        """
        vector_engine = get_vector_engine()

        try:
            found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=self.top_k)
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

        return [result.payload for result in found_chunks]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Generates a completion using document chunks context.

        If the context is not provided, it retrieves the context based on the query. Returns the
        context, which can be used for further processing or generation of outputs.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - context (Optional[Any]): Optional pre-fetched context to use for generating the
              completion; if None, it retrieves the context for the query. (default None)

        Returns:
        --------

            - Any: The context used for the completion or the retrieved context if none was
              provided.
        """
        if context is None:
            context = await self.get_context(query)
        return context
