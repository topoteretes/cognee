import asyncio
import time
from typing import Any, List, Optional, Type

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge

logger = get_logger(level=ERROR)


def format_triplets(edges):
    """Formats edges into human-readable triplet strings."""
    triplets = []
    for edge in edges:
        node1 = edge.node1
        node2 = edge.node2
        edge_attributes = edge.attributes
        node1_attributes = node1.attributes
        node2_attributes = node2.attributes

        # Filter only non-None properties
        node1_info = {key: value for key, value in node1_attributes.items() if value is not None}
        node2_info = {key: value for key, value in node2_attributes.items() if value is not None}
        edge_info = {key: value for key, value in edge_attributes.items() if value is not None}

        # Create the formatted triplet
        triplet = f"Node1: {node1_info}\nEdge: {edge_info}\nNode2: {node2_info}\n\n\n"
        triplets.append(triplet)

    return "".join(triplets)


async def get_memory_fragment(
    properties_to_project: Optional[List[str]] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    relevant_ids_to_filter: Optional[List[str]] = None,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> CogneeGraph:
    """Creates and initializes a CogneeGraph memory fragment with optional property projections."""
    if properties_to_project is None:
        properties_to_project = ["id", "description", "name", "type", "text"]

    memory_fragment = CogneeGraph()

    try:
        graph_engine = await get_graph_engine()
        await memory_fragment.project_graph_from_db(
            graph_engine,
            node_properties_to_project=properties_to_project,
            edge_properties_to_project=["relationship_name", "edge_text"],
            node_type=node_type,
            node_name=node_name,
            relevant_ids_to_filter=relevant_ids_to_filter,
            triplet_distance_penalty=triplet_distance_penalty,
        )
    except EntityNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Error during memory fragment creation: {str(e)}")

    return memory_fragment


class _BruteForceTripletSearchEngine:
    """Internal search engine for brute force triplet search operations."""

    def __init__(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        properties_to_project: Optional[List[str]],
        memory_fragment: Optional[CogneeGraph],
        node_type: Optional[Type],
        node_name: Optional[List[str]],
        wide_search_limit: Optional[int],
        triplet_distance_penalty: float,
    ):
        self.query = query
        self.top_k = top_k
        self.collections = collections
        self.properties_to_project = properties_to_project
        self.memory_fragment = memory_fragment
        self.node_type = node_type
        self.node_name = node_name
        self.wide_search_limit = wide_search_limit
        self.triplet_distance_penalty = triplet_distance_penalty
        self.vector_engine = self._load_vector_engine()
        self.query_vector = None
        self.node_distances = None
        self.edge_distances = None

    async def search(self) -> List[Edge]:
        """Orchestrates the brute force triplet search workflow."""
        await self._embed_query_text()
        await self._retrieve_and_set_vector_distances()

        if not (self.edge_distances or any(self.node_distances.values())):
            return []

        await self._ensure_memory_fragment_is_loaded()
        await self._map_distances_to_memory_fragment()

        return await self.memory_fragment.calculate_top_triplet_importances(k=self.top_k)

    def _load_vector_engine(self):
        """Loads the vector engine instance."""
        try:
            return get_vector_engine()
        except Exception as e:
            logger.error("Failed to initialize vector engine: %s", e)
            raise RuntimeError("Initialization error") from e

    async def _embed_query_text(self):
        """Converts query text into embedding vector."""
        query_embeddings = await self.vector_engine.embedding_engine.embed_text([self.query])
        self.query_vector = query_embeddings[0]

    async def _retrieve_and_set_vector_distances(self):
        """Searches all collections in parallel and sets node/edge distances directly."""
        start_time = time.time()
        search_results = await self._run_parallel_collection_searches()
        elapsed_time = time.time() - start_time

        collections_with_results = sum(1 for result in search_results if result)
        logger.info(
            f"Vector collection retrieval completed: Retrieved distances from "
            f"{collections_with_results} collections in {elapsed_time:.2f}s"
        )

        self.node_distances = {}
        for collection, result in zip(self.collections, search_results):
            if collection == "EdgeType_relationship_name":
                self.edge_distances = result
            else:
                self.node_distances[collection] = result

    async def _run_parallel_collection_searches(self) -> List[List[Any]]:
        """Executes vector searches across all collections concurrently."""
        search_tasks = [
            self._search_single_collection(collection_name) for collection_name in self.collections
        ]
        return await asyncio.gather(*search_tasks)

    async def _search_single_collection(self, collection_name: str):
        """Searches one collection and returns results or empty list if not found."""
        try:
            return await self.vector_engine.search(
                collection_name=collection_name,
                query_vector=self.query_vector,
                limit=self.wide_search_limit,
            )
        except CollectionNotFoundError:
            return []

    async def _ensure_memory_fragment_is_loaded(self):
        """Loads memory fragment if not already provided."""
        if self.memory_fragment is None:
            relevant_node_ids = self._extract_relevant_node_ids_for_filtering()
            self.memory_fragment = await get_memory_fragment(
                properties_to_project=self.properties_to_project,
                node_type=self.node_type,
                node_name=self.node_name,
                relevant_ids_to_filter=relevant_node_ids,
                triplet_distance_penalty=self.triplet_distance_penalty,
            )

    def _extract_relevant_node_ids_for_filtering(self) -> Optional[List[str]]:
        """Extracts unique node IDs from search results to filter graph projection."""
        if self.wide_search_limit is None:
            return None

        relevant_node_ids = {
            str(getattr(scored_node, "id"))
            for score_collection in self.node_distances.values()
            if isinstance(score_collection, (list, tuple))
            for scored_node in score_collection
            if getattr(scored_node, "id", None)
        }
        return list(relevant_node_ids)

    async def _map_distances_to_memory_fragment(self):
        """Maps vector distances to nodes and edges in the memory fragment."""
        await self.memory_fragment.map_vector_distances_to_graph_nodes(
            node_distances=self.node_distances
        )
        await self.memory_fragment.map_vector_distances_to_graph_edges(
            edge_distances=self.edge_distances
        )


async def brute_force_triplet_search(
    query: str,
    top_k: int = 5,
    collections: Optional[List[str]] = None,
    properties_to_project: Optional[List[str]] = None,
    memory_fragment: Optional[CogneeGraph] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> List[Edge]:
    """
    Performs a brute force search to retrieve the top triplets from the graph.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to retrieve.
        collections (Optional[List[str]]): List of collections to query.
        properties_to_project (Optional[List[str]]): List of properties to project.
        memory_fragment (Optional[CogneeGraph]): Existing memory fragment to reuse.
        node_type: node type to filter
        node_name: node name to filter
        wide_search_top_k (Optional[int]): Number of initial elements to retrieve from collections
        triplet_distance_penalty (Optional[float]): Default distance penalty in graph projection

    Returns:
        list: The top triplet results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Setting wide search limit based on the parameters
    non_global_search = node_name is None
    wide_search_limit = wide_search_top_k if non_global_search else None

    if collections is None:
        collections = [
            "Entity_name",
            "TextSummary_text",
            "EntityType_name",
            "DocumentChunk_text",
        ]

    if "EdgeType_relationship_name" not in collections:
        collections.append("EdgeType_relationship_name")

    try:
        engine = _BruteForceTripletSearchEngine(
            query=query,
            top_k=top_k,
            collections=collections,
            properties_to_project=properties_to_project,
            memory_fragment=memory_fragment,
            node_type=node_type,
            node_name=node_name,
            wide_search_limit=wide_search_limit,
            triplet_distance_penalty=triplet_distance_penalty,
        )
        return await engine.search()
    except Exception as error:
        logger.error(
            "Error during brute force search for query: %s. Error: %s",
            query,
            error,
        )
        raise error
