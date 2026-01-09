from typing import List, Optional, Type

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch

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


async def _get_top_triplet_importances(
    memory_fragment: Optional[CogneeGraph],
    vector_search: NodeEdgeVectorSearch,
    properties_to_project: Optional[List[str]],
    node_type: Optional[Type],
    node_name: Optional[List[str]],
    triplet_distance_penalty: float,
    wide_search_limit: Optional[int],
    top_k: int,
) -> List[Edge]:
    """Creates memory fragment (if needed), maps distances, and calculates top triplet importances."""
    if memory_fragment is None:
        relevant_node_ids = vector_search.extract_relevant_node_ids() if wide_search_limit else None
        memory_fragment = await get_memory_fragment(
            properties_to_project=properties_to_project,
            node_type=node_type,
            node_name=node_name,
            relevant_ids_to_filter=relevant_node_ids,
            triplet_distance_penalty=triplet_distance_penalty,
        )

    await memory_fragment.map_vector_distances_to_graph_nodes(
        node_distances=vector_search.node_distances
    )
    await memory_fragment.map_vector_distances_to_graph_edges(
        edge_distances=vector_search.edge_distances
    )

    return await memory_fragment.calculate_top_triplet_importances(k=top_k)


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
        vector_search = NodeEdgeVectorSearch()

        await vector_search.embed_and_retrieve_distances(query, collections, wide_search_limit)

        if not vector_search.has_results():
            return []

        return await _get_top_triplet_importances(
            memory_fragment,
            vector_search,
            properties_to_project,
            node_type,
            node_name,
            triplet_distance_penalty,
            wide_search_limit,
            top_k,
        )
    except Exception as error:
        logger.error(
            "Error during brute force search for query: %s. Error: %s",
            query,
            error,
        )
        raise error
