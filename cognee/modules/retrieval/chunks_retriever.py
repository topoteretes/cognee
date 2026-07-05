from typing import Any, List, Optional, Union

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions.exceptions import (
    CollectionNotFoundError,
)
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

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
        dataset_ids: Optional[list] = None,  # <-- ADDED FOR TENANT ISOLATION
    ):
        """
        Initializes the chunk retriever.
        """
        self.top_k = top_k
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator
        self.dataset_ids = dataset_ids  # <-- STORED HIERARCHICALLY

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Generates a completion using document chunks context.
        In case of the Chunks Retriever, we do not generate a completion, we just return
        the payloads of found chunks.
        """
        if retrieved_objects:
            chunk_payloads = [found_chunk.payload for found_chunk in retrieved_objects]
            return chunk_payloads
        return []

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves context from retrieved chunks, in text form.
        """
        if retrieved_objects:
            chunk_payload_texts = [found_chunk.payload["text"] for found_chunk in retrieved_objects]
            return "\n".join(chunk_payload_texts)
        return ""

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves document chunks context based on the query.
        """
        logger.info(
            f"Starting chunk retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        unified = await get_unified_engine()
        vector_engine = unified.vector

        # SECURITY FIX: Build filter parameter mapping
        search_filter = {"dataset_id": self.dataset_ids} if self.dataset_ids else None

        try:
            found_chunks = await vector_engine.search(
                collection_name="DocumentChunk_text",
                query_text=query,
                limit=self.top_k,
                include_payload=True,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
                query_filter=search_filter,  # <-- INJECTED FILTER PAYLOAD
            )
            logger.info(f"Found {len(found_chunks)} chunks from vector search")

            return found_chunks

        except CollectionNotFoundError as error:
            logger.error("DocumentChunk_text collection not found in vector database")
            raise NoDataError("No data found in the system, please add data first.") from error


# --- Standalone Function Placed Correctly Outside the Class Body ---
async def chunks_retriever(
    query: str, dataset_ids: list = None, top_k: int = 5, config: dict = None
):
    unified_engine = await get_unified_engine()
    vector_engine = unified_engine.vector

    search_filter = {"dataset_id": dataset_ids} if dataset_ids else None

    results = await vector_engine.search(
        collection_name="data_chunks",
        query_text=query,
        query_filter=search_filter,
        limit=top_k,
    )
    return results
