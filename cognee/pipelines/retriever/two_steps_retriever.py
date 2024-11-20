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
    results_dict = delete_duplicated_vector_db_elements(collections, results)# :TODO: Change when vector db is fixed
    # results_dict = {collection: result for collection, result in zip(collections, results)}
    ##############################################

    memory_fragment = CogneeGraph()

    await memory_fragment.project_graph_from_db(graph_engine,
                                          node_properties_to_project=['id',
                                                                      'community'],
                                          edge_properties_to_project=['id',
                                                                      'relationship_name'],
                                          directed=True,
                                          node_dimension=1,
                                          edge_dimension=1,
                                          memory_fragment_filter=[])

    print()


    raise(NotImplementedError)