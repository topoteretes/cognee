import asyncio
import logging
from typing import List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry


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


async def brute_force_triplet_search(
    query: str, user: User = None, top_k=5, collections=None
) -> list:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise PermissionError("No user found in the system. Please create a user.")

    retrieved_results = await brute_force_search(query, user, top_k, collections=collections)
    return retrieved_results


def delete_duplicated_vector_db_elements(
    collections, results
):  #:TODO: This is just for now to fix vector db duplicates
    results_dict = {}
    for collection, results in zip(collections, results):
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.id)
            else:
                print(f"Duplicate found in collection '{collection}': {result.id}")
        results_dict[collection] = unique_results

    return results_dict


async def brute_force_search(
    query: str, user: User, top_k: int, collections: List[str] = None
) -> list:
    """
    Performs a brute force search to retrieve the top triplets from the graph.

    Args:
        query (str): The search query.
        user (User): The user performing the search.
        top_k (int): The number of top results to retrieve.
        collections (Optional[List[str]]): List of collections to query. Defaults to predefined collections.

    Returns:
        list: The top triplet results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if collections is None:
        collections = [
            "entity_name",
            "text_summary_text",
            "entity_type_name",
            "document_chunk_text",
        ]

    try:
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as e:
        logging.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    send_telemetry("cognee.brute_force_triplet_search EXECUTION STARTED", user.id)

    try:
        results = await asyncio.gather(
            *[
                vector_engine.get_distance_from_collection_elements(collection, query_text=query)
                for collection in collections
            ]
        )

        ############################################# :TODO: Change when vector db does not contain duplicates
        node_distances = delete_duplicated_vector_db_elements(collections, results)
        # node_distances = {collection: result for collection, result in zip(collections, results)}
        ##############################################

        memory_fragment = CogneeGraph()

        await memory_fragment.project_graph_from_db(
            graph_engine,
            node_properties_to_project=["id", "description", "name", "type", "text"],
            edge_properties_to_project=["relationship_name"],
        )

        await memory_fragment.map_vector_distances_to_graph_nodes(node_distances=node_distances)

        #:TODO: Change when vectordb contains edge embeddings
        await memory_fragment.map_vector_distances_to_graph_edges(vector_engine, query)

        results = await memory_fragment.calculate_top_triplet_importances(k=top_k)

        send_telemetry("cognee.brute_force_triplet_search EXECUTION STARTED", user.id)

        #:TODO: Once we have Edge pydantic models we should retrieve the exact edge and node objects from graph db
        return results

    except Exception as e:
        logging.error(
            "Error during brute force search for user: %s, query: %s. Error: %s", user.id, query, e
        )
        send_telemetry("cognee.brute_force_triplet_search EXECUTION FAILED", user.id)
        raise RuntimeError("An error occurred during brute force search") from e
