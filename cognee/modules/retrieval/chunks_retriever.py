from typing import Any, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError

logger = get_logger("ChunksRetriever")


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
        logger.info(
            f"Starting chunk retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        vector_engine = get_vector_engine()

        try:
            found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=self.top_k)
            logger.info(f"Found {len(found_chunks)} chunks from vector search")
        except CollectionNotFoundError as error:
            logger.error("DocumentChunk_text collection not found in vector database")
            raise NoDataError("No data found in the system, please add data first.") from error

        chunk_payloads = [result.payload for result in found_chunks]
        logger.info(f"Returning {len(chunk_payloads)} chunk payloads")
        return chunk_payloads

    async def get_completion(
        self, query: str, context: Optional[Any] = None, session_id: Optional[str] = None
    ) -> Any:
        """
        Generates a completion using document chunks context.

        If the context is not provided, it retrieves the context based on the query. Returns the
        context, which can be used for further processing or generation of outputs.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - context (Optional[Any]): Optional pre-fetched context to use for generating the
              completion; if None, it retrieves the context for the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: The context used for the completion or the retrieved context if none was
              provided.
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
