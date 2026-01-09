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

        node1_info = {key: value for key, value in node1_attributes.items() if value is not None}
        node2_info = {key: value for key, value in node2_attributes.items() if value is not None}
        edge_info = {key: value for key, value in edge_attributes.items() if value is not None}

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


class TripletSearchContext:
    """Pure state container for triplet search operations."""

    def __init__(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        properties_to_project: Optional[List[str]],
        node_type: Optional[Type],
        node_name: Optional[List[str]],
        wide_search_limit: Optional[int],
        triplet_distance_penalty: float,
    ):
        self.query = query
        self.top_k = top_k
        self.collections = collections
        self.properties_to_project = properties_to_project
        self.node_type = node_type
        self.node_name = node_name
        self.wide_search_limit = wide_search_limit
        self.triplet_distance_penalty = triplet_distance_penalty

        self.query_vector = None
        self.node_distances = None
        self.edge_distances = None

    def has_results(self) -> bool:
        """Checks if any collections returned results."""
        return bool(self.edge_distances or any(self.node_distances.values()))

    def extract_relevant_node_ids(self) -> Optional[List[str]]:
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

    def set_distances_from_results(self, search_results: List[List[Any]]):
        """Separates search results into node and edge distances."""
        self.node_distances = {}
        for collection, result in zip(self.collections, search_results):
            if collection == "EdgeType_relationship_name":
                self.edge_distances = result
            else:
                self.node_distances[collection] = result


async def _search_single_collection(
    vector_engine: Any, search_context: TripletSearchContext, collection_name: str
):
    """Searches one collection and returns results or empty list if not found."""
    try:
        return await vector_engine.search(
            collection_name=collection_name,
            query_vector=search_context.query_vector,
            limit=search_context.wide_search_limit,
        )
    except CollectionNotFoundError:
        return []


async def _embed_and_retrieve_distances(search_context: TripletSearchContext):
    """Embeds query and retrieves vector distances from all collections."""
    vector_engine = get_vector_engine()

    query_embeddings = await vector_engine.embedding_engine.embed_text([search_context.query])
    search_context.query_vector = query_embeddings[0]

    start_time = time.time()
    search_tasks = [
        _search_single_collection(vector_engine, search_context, collection)
        for collection in search_context.collections
    ]
    search_results = await asyncio.gather(*search_tasks)

    elapsed_time = time.time() - start_time
    collections_with_results = sum(1 for result in search_results if result)
    logger.info(
        f"Vector collection retrieval completed: Retrieved distances from "
        f"{collections_with_results} collections in {elapsed_time:.2f}s"
    )

    search_context.set_distances_from_results(search_results)


async def _create_memory_fragment(search_context: TripletSearchContext) -> CogneeGraph:
    """Creates memory fragment using search context properties."""
    relevant_node_ids = search_context.extract_relevant_node_ids()
    return await get_memory_fragment(
        properties_to_project=search_context.properties_to_project,
        node_type=search_context.node_type,
        node_name=search_context.node_name,
        relevant_ids_to_filter=relevant_node_ids,
        triplet_distance_penalty=search_context.triplet_distance_penalty,
    )


async def _map_distances_to_fragment(
    search_context: TripletSearchContext, memory_fragment: CogneeGraph
):
    """Maps vector distances from search context to memory fragment."""
    await memory_fragment.map_vector_distances_to_graph_nodes(
        node_distances=search_context.node_distances
    )
    await memory_fragment.map_vector_distances_to_graph_edges(
        edge_distances=search_context.edge_distances
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

    wide_search_limit = wide_search_top_k if node_name is None else None

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
        search_context = TripletSearchContext(
            query=query,
            top_k=top_k,
            collections=collections,
            properties_to_project=properties_to_project,
            node_type=node_type,
            node_name=node_name,
            wide_search_limit=wide_search_limit,
            triplet_distance_penalty=triplet_distance_penalty,
        )

        await _embed_and_retrieve_distances(search_context)

        if not search_context.has_results():
            return []

        if memory_fragment is None:
            memory_fragment = await _create_memory_fragment(search_context)

        await _map_distances_to_fragment(search_context, memory_fragment)

        return await memory_fragment.calculate_top_triplet_importances(k=search_context.top_k)
    except Exception as error:
        logger.error(
            "Error during brute force search for query: %s. Error: %s",
            query,
            error,
        )
        raise error
