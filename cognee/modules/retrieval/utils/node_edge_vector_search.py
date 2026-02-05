import asyncio
import time
from typing import Any, List, Optional

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector import get_vector_engine

logger = get_logger(level=ERROR)


class NodeEdgeVectorSearch:
    """Manages vector search and distance retrieval for graph nodes and edges."""

    def __init__(self, edge_collection: str = "EdgeType_relationship_name", vector_engine=None):
        self.edge_collection = edge_collection
        self.vector_engine = vector_engine or self._init_vector_engine()
        self.query_vector: Optional[Any] = None
        self.node_distances: dict[str, list[Any]] = {}
        self.edge_distances: list[Any] = []
        self.query_list_length: Optional[int] = None

    def _init_vector_engine(self):
        try:
            return get_vector_engine()
        except Exception as e:
            logger.error("Failed to initialize vector engine: %s", e)
            raise RuntimeError("Initialization error") from e

    async def embed_and_retrieve_distances(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        collections: List[str] = None,
        wide_search_limit: Optional[int] = None,
    ):
        """Embeds query/queries and retrieves vector distances from all collections."""
        if query is not None and query_batch is not None:
            raise ValueError("Cannot provide both 'query' and 'query_batch'; use exactly one.")
        if query is None and query_batch is None:
            raise ValueError("Must provide either 'query' or 'query_batch'.")
        if not collections:
            raise ValueError("'collections' must be a non-empty list.")

        start_time = time.time()

        if query_batch is not None:
            self.query_list_length = len(query_batch)
            search_results = await self._run_batch_search(collections, query_batch)
        else:
            self.query_list_length = None
            search_results = await self._run_single_search(collections, query, wide_search_limit)

        elapsed_time = time.time() - start_time
        collections_with_results = sum(1 for result in search_results if any(result))
        logger.info(
            f"Vector collection retrieval completed: Retrieved distances from "
            f"{collections_with_results} collections in {elapsed_time:.2f}s"
        )

        self.set_distances_from_results(collections, search_results, self.query_list_length)

    def has_results(self) -> bool:
        """Checks if any collections returned results."""
        if self.query_list_length is None:
            if self.edge_distances and any(self.edge_distances):
                return True
            return any(
                bool(collection_results) for collection_results in self.node_distances.values()
            )

        if self.edge_distances and any(inner_list for inner_list in self.edge_distances):
            return True
        return any(
            any(results_per_query for results_per_query in collection_results)
            for collection_results in self.node_distances.values()
        )

    def extract_relevant_node_ids(self) -> List[str]:
        """Extracts unique node IDs from search results."""
        if self.query_list_length is not None:
            return []
        relevant_node_ids = set()
        for scored_results in self.node_distances.values():
            for scored_node in scored_results:
                node_id = getattr(scored_node, "id", None)
                if node_id:
                    relevant_node_ids.add(str(node_id))
        return list(relevant_node_ids)

    def set_distances_from_results(
        self,
        collections: List[str],
        search_results: List[List[Any]],
        query_list_length: Optional[int] = None,
    ):
        """Separates search results into node and edge distances with stable shapes.

        Ensures all collections are present in the output, even if empty:
        - Batch mode: missing/empty collections become [[]] * query_list_length
        - Single mode: missing/empty collections become []
        """
        self.node_distances = {}
        self.edge_distances = (
            [] if query_list_length is None else [[] for _ in range(query_list_length)]
        )
        for collection, result in zip(collections, search_results):
            if not result:
                empty_result = (
                    [] if query_list_length is None else [[] for _ in range(query_list_length)]
                )
                if collection == self.edge_collection:
                    self.edge_distances = empty_result
                else:
                    self.node_distances[collection] = empty_result
            else:
                if collection == self.edge_collection:
                    self.edge_distances = result
                else:
                    self.node_distances[collection] = result

    async def _run_batch_search(
        self, collections: List[str], query_batch: List[str]
    ) -> List[List[Any]]:
        """Runs batch search across all collections and returns list-of-lists per collection."""
        search_tasks = [
            self._search_batch_collection(collection, query_batch) for collection in collections
        ]
        return await asyncio.gather(*search_tasks)

    async def _search_batch_collection(
        self, collection_name: str, query_batch: List[str]
    ) -> List[List[Any]]:
        """Searches one collection with batch queries and returns list-of-lists."""
        try:
            return await self.vector_engine.batch_search(
                collection_name=collection_name, query_texts=query_batch, limit=None
            )
        except CollectionNotFoundError:
            return [[]] * len(query_batch)

    async def _run_single_search(
        self, collections: List[str], query: str, wide_search_limit: Optional[int]
    ) -> List[List[Any]]:
        """Runs single query search and returns flat lists per collection.

        Returns a list where each element is a collection's results (flat list).
        These are stored as flat lists in node_distances/edge_distances for single-query mode.
        """
        await self._embed_query(query)
        search_tasks = [
            self._search_single_collection(self.vector_engine, wide_search_limit, collection)
            for collection in collections
        ]
        search_results = await asyncio.gather(*search_tasks)
        return search_results

    async def _embed_query(self, query: str):
        """Embeds the query and stores the resulting vector."""
        query_embeddings = await self.vector_engine.embedding_engine.embed_text([query])
        self.query_vector = query_embeddings[0]

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
