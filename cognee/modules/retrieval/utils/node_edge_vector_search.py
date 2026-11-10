import asyncio
import time
from typing import Any

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.modules.observability import COGNEE_VECTOR_COLLECTION, new_span
from cognee.shared.logging_utils import ERROR, get_logger

logger = get_logger(level=ERROR)


def engine_supports_id_scoring(engine) -> bool:
    """Whether ``engine`` implements ``score_by_ids`` rather than inheriting the stub."""
    # Engine leases wrap bound methods, but preserve the adapter's __class__.
    if getattr(engine.__class__, "score_by_ids", None) is VectorDBInterface.score_by_ids:
        return False
    score_by_ids = getattr(engine, "score_by_ids", None)
    return (
        callable(score_by_ids)
        and getattr(score_by_ids, "__func__", None) is not VectorDBInterface.score_by_ids
    )


class NodeEdgeVectorSearch:
    """Manages vector search and distance retrieval for graph nodes and edges."""

    def __init__(self, edge_collection: str = "EdgeType_relationship_name", vector_engine=None):
        self.edge_collection = edge_collection
        # ``get_vector_engine_async()`` is async, so this sync ``__init__`` can't
        # eagerly resolve it. Keep the (possibly-None) injected engine and
        # resolve lazily in the first async method via ``_get_vector_engine()``.
        self.vector_engine = vector_engine
        self.query_vector: Any | None = None
        self.node_distances: dict[str, list[Any]] = {}
        self.edge_distances: list[Any] = []
        self.query_list_length: int | None = None

    async def _get_vector_engine(self):
        if self.vector_engine is None:
            try:
                self.vector_engine = await get_vector_engine_async()
            except Exception as e:
                logger.error("Failed to initialize vector engine: %s", e)
                raise RuntimeError("Initialization error") from e
        return self.vector_engine

    async def supports_id_scoring(self) -> bool:
        return engine_supports_id_scoring(await self._get_vector_engine())

    async def embed_and_retrieve_distances(
        self,
        query: str | None = None,
        query_batch: list[str] | None = None,
        collections: list[str] | None = None,
        wide_search_limit: int | None = None,
        node_name: list[str] | None = None,
        node_name_filter_operator: str = "OR",
        ids_by_collection: dict[str, list[str]] | None = None,
    ):
        """Embed queries and retrieve distances, optionally restricted to exact graph IDs."""
        if query is not None and query_batch is not None:
            raise ValueError("Cannot provide both 'query' and 'query_batch'; use exactly one.")
        if query is None and query_batch is None:
            raise ValueError("Must provide either 'query' or 'query_batch'.")
        if not collections:
            raise ValueError("'collections' must be a non-empty list.")

        with new_span("cognee.retrieval.vector_search") as span:
            span.set_attribute("cognee.vector.collection_count", len(collections))
            span.set_attribute(COGNEE_VECTOR_COLLECTION, ", ".join(collections))
            span.set_attribute(
                "cognee.vector.mode", "batch" if query_batch is not None else "single"
            )
            if wide_search_limit is not None:
                span.set_attribute("cognee.vector.wide_search_limit", wide_search_limit)

            start_time = time.time()

            if ids_by_collection is not None:
                self.query_list_length = len(query_batch) if query_batch is not None else None
                search_results = await self._run_id_search(
                    collections, query, query_batch, ids_by_collection
                )
            elif query_batch is not None:
                self.query_list_length = len(query_batch)
                span.set_attribute("cognee.vector.batch_size", len(query_batch))
                search_results = await self._run_batch_search(collections, query_batch)
            else:
                self.query_list_length = None
                search_results = await self._run_single_search(
                    collections, query, wide_search_limit, node_name, node_name_filter_operator
                )

            elapsed_time = time.time() - start_time
            collections_with_results = sum(1 for result in search_results if any(result))
            logger.info(
                f"Vector collection retrieval completed: Retrieved distances from "
                f"{collections_with_results} collections in {elapsed_time:.2f}s"
            )

            span.set_attribute("cognee.vector.collections_with_results", collections_with_results)
            span.set_attribute("cognee.vector.duration_ms", round(elapsed_time * 1000, 1))

            self.set_distances_from_results(collections, search_results, self.query_list_length)

    async def _run_id_search(
        self,
        collections: list[str],
        query: str | None,
        query_batch: list[str] | None,
        ids_by_collection: dict[str, list[str]],
    ) -> list[list[Any]]:
        engine = await self._get_vector_engine()
        if query_batch is None:
            await self._embed_query(query)
            query_vectors = [self.query_vector]
        else:
            self.query_vector = None
            query_vectors = await engine.embedding_engine.embed_text(query_batch)

        async def score_collection(collection):
            ids = ids_by_collection.get(collection, [])
            if not ids:
                return [[] for _ in query_vectors] if query_batch is not None else []
            try:
                if query_batch is None:
                    return await engine.score_by_ids(
                        collection_name=collection,
                        data_point_ids=ids,
                        query_vector=query_vectors[0],
                    )
                # One adapter call scores the whole batch: the adapter decides whether
                # that is concurrent per-vector scoring or a single filtered scan.
                return await engine.score_by_ids_batch(
                    collection_name=collection, data_point_ids=ids, query_vectors=query_vectors
                )
            except CollectionNotFoundError:
                return [[] for _ in query_vectors] if query_batch is not None else []

        return await asyncio.gather(*(score_collection(collection) for collection in collections))

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

    def extract_relevant_node_ids(self) -> list[str]:
        """Extracts unique node IDs from search results, nearest first.

        Scores are vector distances (lower = closer). A node found in several
        collections is ranked by its best score, so callers can take a top-k
        slice (e.g. neighborhood seeds) and get the most similar nodes.
        """
        if self.query_list_length is not None:
            return []
        best_score: dict[str, float] = {}
        for scored_results in self.node_distances.values():
            for scored_node in scored_results:
                node_id = getattr(scored_node, "id", None)
                if not node_id:
                    continue
                score = getattr(scored_node, "score", None)
                score = float("inf") if score is None else score
                node_id = str(node_id)
                if node_id not in best_score or score < best_score[node_id]:
                    best_score[node_id] = score
        return sorted(best_score, key=best_score.__getitem__)

    def set_distances_from_results(
        self,
        collections: list[str],
        search_results: list[list[Any]],
        query_list_length: int | None = None,
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
        self, collections: list[str], query_batch: list[str]
    ) -> list[list[Any]]:
        """Runs batch search across all collections and returns list-of-lists per collection."""
        vector_engine = await self._get_vector_engine()
        existing = set(await vector_engine.existing_collections(collections))
        search_tasks = [
            self._search_batch_collection(collection, query_batch)
            for collection in collections
            if collection in existing
        ]
        found = iter(await asyncio.gather(*search_tasks))
        return [
            next(found) if collection in existing else [[]] * len(query_batch)
            for collection in collections
        ]

    async def _search_batch_collection(
        self, collection_name: str, query_batch: list[str]
    ) -> list[list[Any]]:
        """Searches one collection with batch queries and returns list-of-lists."""
        try:
            vector_engine = await self._get_vector_engine()
            return await vector_engine.batch_search(
                collection_name=collection_name, query_texts=query_batch, limit=None
            )
        except CollectionNotFoundError:
            return [[]] * len(query_batch)

    async def _run_single_search(
        self,
        collections: list[str],
        query: str,
        wide_search_limit: int | None,
        node_name: list[str] | None,
        node_name_filter_operator: str,
    ) -> list[list[Any]]:
        """Runs single query search and returns flat lists per collection.

        Returns a list where each element is a collection's results (flat list).
        These are stored as flat lists in node_distances/edge_distances for single-query mode.
        """
        await self._embed_query(query)
        vector_engine = await self._get_vector_engine()
        # Missing collections answer empty without a search: on pgvector each
        # one would otherwise cost a connection to find out it does not exist.
        existing = set(await vector_engine.existing_collections(collections))
        search_tasks = [
            self._search_single_collection(
                vector_engine,
                wide_search_limit,
                collection,
                node_name,
                node_name_filter_operator,
            )
            for collection in collections
            if collection in existing
        ]
        found = iter(await asyncio.gather(*search_tasks))
        return [next(found) if collection in existing else [] for collection in collections]

    async def _embed_query(self, query: str):
        """Embeds the query and stores the resulting vector."""
        with new_span("cognee.retrieval.embed_query") as span:
            span.set_attribute("cognee.vector.query_length", len(query))
            vector_engine = await self._get_vector_engine()
            query_embeddings = await vector_engine.embedding_engine.embed_text([query])
            self.query_vector = query_embeddings[0]
            span.set_attribute("cognee.vector.embedding_dimensions", len(self.query_vector))

    async def _search_single_collection(
        self,
        vector_engine: Any,
        wide_search_limit: int | None,
        collection_name: str,
        node_name: list[str] | None,
        node_name_filter_operator: str,
    ):
        """Searches one collection and returns results or empty list if not found."""
        if collection_name == self.edge_collection:
            # Edge types are shared vocabulary, not node-set members: they only score
            # edges the graph projection already scoped, so the node filter never
            # applies to them (the hybrid retriever does the same).
            node_name = None
        try:
            return await vector_engine.search(
                collection_name=collection_name,
                query_vector=self.query_vector,
                limit=wide_search_limit,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
            )
        except CollectionNotFoundError:
            return []
