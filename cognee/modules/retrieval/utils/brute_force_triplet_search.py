import asyncio
from typing import List, Optional

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

logger = get_logger(level=ERROR)


def format_triplets(edges):
    print("\n\n\n")

    def filter_attributes(obj, attributes):
        """Helper function to filter out non-None properties, including nested dicts."""
        result = {}
        for attr in attributes:
            value = getattr(obj, attr, None)
            if value is not None:
                # If the value is a dict, extract relevant keys from it
                if isinstance(value, dict):
                    nested_values = {
                        k: v for k, v in value.items() if k in attributes and v is not None
                    }
                    result[attr] = nested_values
                else:
                    result[attr] = value
        return result

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
) -> CogneeGraph:
    """Creates and initializes a CogneeGraph memory fragment with optional property projections."""
    graph_engine = await get_graph_engine()
    memory_fragment = CogneeGraph()

    if properties_to_project is None:
        properties_to_project = ["id", "description", "name", "type", "text"]

    try:
        await memory_fragment.project_graph_from_db(
            graph_engine,
            node_properties_to_project=properties_to_project,
            edge_properties_to_project=["relationship_name"],
        )
    except EntityNotFoundError:
        pass

    return memory_fragment


async def brute_force_triplet_search(
    query: str,
    user: User = None,
    top_k: int = 5,
    collections: List[str] = None,
    properties_to_project: List[str] = None,
    memory_fragment: Optional[CogneeGraph] = None,
) -> list:
    if user is None:
        user = await get_default_user()

    retrieved_results = await brute_force_search(
        query,
        user,
        top_k,
        collections=collections,
        properties_to_project=properties_to_project,
        memory_fragment=memory_fragment,
    )
    return retrieved_results


async def brute_force_search(
    query: str,
    user: User,
    top_k: int,
    collections: List[str] = None,
    properties_to_project: List[str] = None,
    memory_fragment: Optional[CogneeGraph] = None,
) -> list:
    """
    Performs a brute force search to retrieve the top triplets from the graph.

    Args:
        query (str): The search query.
        user (User): The user performing the search.
        top_k (int): The number of top results to retrieve.
        collections (Optional[List[str]]): List of collections to query.
        properties_to_project (Optional[List[str]]): List of properties to project.
        memory_fragment (Optional[CogneeGraph]): Existing memory fragment to reuse.

    Returns:
        list: The top triplet results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if memory_fragment is None:
        memory_fragment = await get_memory_fragment(properties_to_project)

    if collections is None:
        collections = [
            "Entity_name",
            "TextSummary_text",
            "EntityType_name",
            "DocumentChunk_text",
        ]

    try:
        vector_engine = get_vector_engine()
    except Exception as e:
        logger.error("Failed to initialize vector engine: %s", e)
        raise RuntimeError("Initialization error") from e

    send_telemetry("cognee.brute_force_triplet_search EXECUTION STARTED", user.id)

    async def search_in_collection(collection_name: str):
        try:
            return await vector_engine.search(
                collection_name=collection_name, query_text=query, limit=0
            )
        except CollectionNotFoundError:
            return []

    try:
        results = await asyncio.gather(
            *[search_in_collection(collection_name) for collection_name in collections]
        )

        if all(not item for item in results):
            return []

        node_distances = {collection: result for collection, result in zip(collections, results)}

        await memory_fragment.map_vector_distances_to_graph_nodes(node_distances=node_distances)
        await memory_fragment.map_vector_distances_to_graph_edges(vector_engine, query)

        results = await memory_fragment.calculate_top_triplet_importances(k=top_k)

        send_telemetry("cognee.brute_force_triplet_search EXECUTION COMPLETED", user.id)

        return results

    except CollectionNotFoundError:
        return []
    except Exception as error:
        logger.error(
            "Error during brute force search for user: %s, query: %s. Error: %s",
            user.id,
            query,
            error,
        )
        send_telemetry(
            "cognee.brute_force_triplet_search EXECUTION FAILED", user.id, {"error": str(error)}
        )
        raise error
