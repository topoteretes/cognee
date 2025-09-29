import asyncio
import time
from typing import List, Optional, Type

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
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
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
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
            edge_properties_to_project=["relationship_name"],
            node_type=node_type,
            node_name=node_name,
        )

    except EntityNotFoundError:
        # This is expected behavior - continue with empty fragment
        pass
    except Exception as e:
        logger.error(f"Error during memory fragment creation: {str(e)}")
        # Still return the fragment even if projection failed
        pass

    return memory_fragment


async def brute_force_triplet_search(
    query: str,
    top_k: int = 5,
    collections: Optional[List[str]] = None,
    properties_to_project: Optional[List[str]] = None,
    memory_fragment: Optional[CogneeGraph] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
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

    Returns:
        list: The top triplet results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if memory_fragment is None:
        memory_fragment = await get_memory_fragment(
            properties_to_project, node_type=node_type, node_name=node_name
        )

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

    query_vector = (await vector_engine.embedding_engine.embed_text([query]))[0]

    async def search_in_collection(collection_name: str):
        try:
            return await vector_engine.search(
                collection_name=collection_name, query_vector=query_vector, limit=None
            )
        except CollectionNotFoundError:
            return []

    try:
        start_time = time.time()

        results = await asyncio.gather(
            *[search_in_collection(collection_name) for collection_name in collections]
        )

        if all(not item for item in results):
            return []

        # Final statistics
        projection_time = time.time() - start_time
        logger.info(
            f"Vector collection retrieval completed: Retrieved distances from {sum(1 for res in results if res)} collections in {projection_time:.2f}s"
        )

        node_distances = {collection: result for collection, result in zip(collections, results)}

        edge_distances = node_distances.get("EdgeType_relationship_name", None)

        await memory_fragment.map_vector_distances_to_graph_nodes(node_distances=node_distances)
        await memory_fragment.map_vector_distances_to_graph_edges(
            vector_engine=vector_engine, query_vector=query_vector, edge_distances=edge_distances
        )

        results = await memory_fragment.calculate_top_triplet_importances(k=top_k)

        return results

    except CollectionNotFoundError:
        return []
    except Exception as error:
        logger.error(
            "Error during brute force search for query: %s. Error: %s",
            query,
            error,
        )
        raise error
