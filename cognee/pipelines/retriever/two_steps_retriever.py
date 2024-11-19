import asyncio
from uuid import UUID
from enum import Enum
from typing import Callable, Dict
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine


async def two_step_retriever(query: Dict[str, str], user: User = None) -> list:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise PermissionError("No user found in the system. Please create a user.")

    own_document_ids = await get_document_ids_for_user(user.id)
    retrieved_results = await run_two_step_retriever(query, user)

    filtered_search_results = []


    return retrieved_results


async def run_two_step_retriever(query: str, user, community_filter = []) -> list:
    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    collections = ["Entity_name", "TextSummary_text", 'EntityType_name', 'DocumentChunk_text']
    results = await asyncio.gather(
        *[vector_engine.get_distances_of_collection(collection, query_text=query) for collection in collections]
    )

    ############################################# This part is a quick fix til we don't fix the vector db inconsistency
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
    # :TODO: Due to duplicates and inconsistent vector db state now am collecting
    # :TODO: the first appearance of the object but this code should be the solution once the db is fixed.
    # results_dict = {collection: result for collection, result in zip(collections, results)}
    ##############################################

    print()


    raise(NotImplementedError)