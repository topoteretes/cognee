from typing import Any, Optional, List, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.chunk_context_enrichment import enrich_chunk_results
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
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        expand_neighbors: int = 0,
        strict_enrichment: bool = False,
    ):
        """
        Initializes the chunk retriever.

        Parameters:
        -----------

            - top_k (Optional[int]): Maximum number of chunks to retrieve.
              Defaults to 5.
            - node_name (Optional[List[str]]): Node names used to filter chunks by
              their belongs_to_set relationship. Defaults to None, which applies no
              node set filtering.
            - node_name_filter_operator (str): Logical operator used when applying
              multiple node_name filters, such as "OR" or "AND". Defaults to "OR".
            - expand_neighbors (int): When > 0, attach neighboring chunks from the
              same parent document to each result (max 10).
            - strict_enrichment (bool): Raise when graph enrichment cannot resolve
              parent documents for retrieved chunks.
        """
        self.top_k = top_k
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator
        self.expand_neighbors = max(0, min(int(expand_neighbors or 0), 10))
        self.strict_enrichment = strict_enrichment

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Generates a completion using document chunks context.
        In case of the Chunks Retriever, we do not generate a completion, we just return
        the payloads of found chunks.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - retrieved_objects (Any): The retrieved objects to be used for generating a completion.
            - context (Any): The context to be used for generating a completion.

        Returns:
        --------

            - List[dict]: A list of payloads of found chunks.
        """
        # TODO: Do we want to generate a completion using LLM here?
        if retrieved_objects:
            chunk_payloads = [found_chunk.payload for found_chunk in retrieved_objects]
            return chunk_payloads
        else:
            return []

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves context from retrieved chunks, in text form.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant document chunks.
            - retrieved_objects (Any): The retrieved objects to be used for generating textual context.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved chunks, or an
              empty string if none are found.
        """
        if retrieved_objects:
            chunk_payload_texts = [found_chunk.payload["text"] for found_chunk in retrieved_objects]
            return "\n".join(chunk_payload_texts)
        else:
            return ""

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves document chunks context based on the query.
        Searches for document chunks relevant to the specified query using a vector engine.
        Raises a NoDataError if no data is found in the system.
        Parameters:
        -----------
            - query (str): The query string to search for relevant document chunks.
        Returns:
        --------
            - Any: A list of document chunks retrieved from the search.
        """
        logger.info(
            f"Starting chunk retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        unified = await get_unified_engine()
        vector_engine = unified.vector

        try:
            found_chunks = await vector_engine.search(
                "DocumentChunk_text",
                query,
                limit=self.top_k,
                include_payload=True,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
            )
            logger.info(f"Found {len(found_chunks)} chunks from vector search")
        except CollectionNotFoundError as error:
            logger.error("DocumentChunk_text collection not found in vector database")
            raise NoDataError("No data found in the system, please add data first.") from error

        if not found_chunks:
            return found_chunks

        try:
            graph_engine = unified.graph
        except Exception as error:
            message = f"Graph engine unavailable: {error}"
            if self.strict_enrichment:
                raise NoDataError(message) from error
            logger.warning("%s, skipping enrichment", message)
            return found_chunks

        try:
            return await enrich_chunk_results(
                graph_engine,
                found_chunks,
                expand_neighbors=self.expand_neighbors,
                strict_enrichment=self.strict_enrichment,
            )
        except ValueError as error:
            if self.strict_enrichment:
                raise NoDataError(str(error)) from error
            logger.warning("Chunk enrichment skipped: %s", error)
            return found_chunks
