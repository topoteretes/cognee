from uuid import UUID
from enum import Enum
from typing import Callable, Dict
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.tasks.chunks import query_chunks
from cognee.tasks.graph import query_graph_connections
from cognee.tasks.summarization import query_summaries

class SearchType(Enum):
    SUMMARIES = "SUMMARIES"
    INSIGHTS = "INSIGHTS"
    CHUNKS = "CHUNKS"

async def search(search_type: SearchType, query: str, user: User = None) -> list:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise PermissionError("No user found in the system. Please create a user.")

    own_document_ids = await get_document_ids_for_user(user.id)
    search_results = await specific_search(search_type, query, user)

    filtered_search_results = []

    for search_result in search_results:
        document_id = search_result["document_id"] if "document_id" in search_result else None
        document_id = UUID(document_id) if type(document_id) == str else document_id

        if document_id is None or document_id in own_document_ids:
            filtered_search_results.append(search_result)

    return filtered_search_results

async def specific_search(search_type: SearchType, query: str, user) -> list:
    search_tasks: Dict[SearchType, Callable] = {
        SearchType.SUMMARIES: query_summaries,
        SearchType.INSIGHTS: query_graph_connections,
        SearchType.CHUNKS: query_chunks,
    }

    search_task = search_tasks.get(search_type)

    if search_task is None:
        raise ValueError(f"Unsupported search type: {search_type}")

    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    results = await search_task(query)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return results
