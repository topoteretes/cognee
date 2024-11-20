import asyncio
from uuid import UUID
from enum import Enum
from typing import Callable, Dict
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from openai import organization
from sympy.codegen.fnodes import dimension


def format_triplets(edges):
    def filter_attributes(obj, attributes):
        """Helper function to filter out non-None properties, including nested dicts."""
        print("\n\n\n")
        result = {}
        for attr in attributes:
            value = getattr(obj, attr, None)
            if value is not None:
                # If the value is a dict, extract relevant keys from it
                if isinstance(value, dict):
                    nested_values = {k: v for k, v in value.items() if k in attributes and v is not None}
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
        triplet = (
            f"Node1: {node1_info}\n"
            f"Edge: {edge_info}\n"
            f"Node2: {node2_info}\n\n\n"  # Add three blank lines for separation
        )
        triplets.append(triplet)

    return "".join(triplets)


async def two_step_retriever(query: Dict[str, str], user: User = None) -> list:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise PermissionError("No user found in the system. Please create a user.")

    own_document_ids = await get_document_ids_for_user(user.id)
    retrieved_results = await run_two_step_retriever(query, user)

    filtered_search_results = []

    return retrieved_results


def delete_duplicated_vector_db_elements(collections, results): #:TODO: This is just for now to fix vector db duplicates
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


async def run_two_step_retriever(query: str, user, community_filter = []) -> list:
    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    collections = ["Entity_name", "TextSummary_text", 'EntityType_name', 'DocumentChunk_text']
    results = await asyncio.gather(
        *[vector_engine.get_distances_of_collection(collection, query_text=query) for collection in collections]
    )

    ############################################# This part is a quick fix til we don't fix the vector db inconsistency
    node_distances = delete_duplicated_vector_db_elements(collections, results)# :TODO: Change when vector db is fixed
    # results_dict = {collection: result for collection, result in zip(collections, results)}
    ##############################################

    memory_fragment = CogneeGraph()

    await memory_fragment.project_graph_from_db(graph_engine,
                                          node_properties_to_project=['id',
                                                                      'description',
                                                                      'name',
                                                                      'type',
                                                                      'text'],
                                          edge_properties_to_project=['id',
                                                                      'relationship_name'])

    await memory_fragment.map_vector_distances_to_graph_nodes(node_distances=node_distances)

    await memory_fragment.map_vector_distances_to_graph_edges(vector_engine, query)# :TODO: This should be coming from vector db

    results = await memory_fragment.calculate_top_triplet_importances(k=5)


    print(format_triplets(results))
    print(f'Query was the following:{query}' )
