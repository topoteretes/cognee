import asyncio
import time
from typing import Any, List, Optional

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector import get_vector_engine

logger = get_logger(level=ERROR)


class NodeEdgeVectorSearch:
    """Manages vector search and distance retrieval for graph nodes and edges."""

    def __init__(self, edge_collection: str = "EdgeType_relationship_name"):
        self.edge_collection = edge_collection
        self.query_vector: Optional[Any] = None
        self.node_distances: dict[str, list[Any]] = {}
        self.edge_distances: Optional[list[Any]] = None

    def has_results(self) -> bool:
        """Checks if any collections returned results."""
        return bool(self.edge_distances) or any(self.node_distances.values())

    def set_distances_from_results(self, collections: List[str], search_results: List[List[Any]]):
        """Separates search results into node and edge distances."""
        self.node_distances = {}
        for collection, result in zip(collections, search_results):
            if collection == self.edge_collection:
                self.edge_distances = result
            else:
                self.node_distances[collection] = result

    def extract_relevant_node_ids(self) -> List[str]:
        """Extracts unique node IDs from search results."""
        relevant_node_ids = {
            str(getattr(scored_node, "id"))
            for score_collection in self.node_distances.values()
            if isinstance(score_collection, (list, tuple))
            for scored_node in score_collection
            if getattr(scored_node, "id", None)
        }
        return list(relevant_node_ids)

    async def embed_and_retrieve_distances(
        self, query: str, collections: List[str], wide_search_limit: Optional[int]
    ):
        """Embeds query and retrieves vector distances from all collections."""
        vector_engine = get_vector_engine()

        query_embeddings = await vector_engine.embedding_engine.embed_text([query])
        self.query_vector = query_embeddings[0]

        start_time = time.time()
        search_tasks = [
            self._search_single_collection(vector_engine, wide_search_limit, collection)
            for collection in collections
        ]
        search_results = await asyncio.gather(*search_tasks)

        elapsed_time = time.time() - start_time
        collections_with_results = sum(1 for result in search_results if result)
        logger.info(
            f"Vector collection retrieval completed: Retrieved distances from "
            f"{collections_with_results} collections in {elapsed_time:.2f}s"
        )

        self.set_distances_from_results(collections, search_results)

    async def _search_single_collection(
        self, vector_engine: Any, wide_search_limit: Optional[int], collection_name: str
    ):
        """Searches one collection and returns results or empty list if not found."""
        try:
            return await vector_engine.search(
                collection_name=collection_name,
                query_vector=self.query_vector,
                limit=wide_search_limit,
            )
        except CollectionNotFoundError:
            return []
