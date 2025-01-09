import json
from uuid import UUID
from enum import Enum
from typing import Callable, Dict, Union

from cognee.exceptions import InvalidValueError
from cognee.modules.search.operations import log_query, log_result
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.utils import send_telemetry
from cognee.modules.users.exceptions import UserNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.tasks.chunks import query_chunks
from cognee.tasks.graph import query_graph_connections
from cognee.tasks.summarization import query_summaries
from cognee.tasks.completion import query_completion
from cognee.tasks.completion import graph_query_completion


class SearchType(Enum):
    SUMMARIES = "SUMMARIES"
    INSIGHTS = "INSIGHTS"
    CHUNKS = "CHUNKS"
    COMPLETION = "COMPLETION"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"


async def search(
    query_type: SearchType,
    query_text: str,
    user: User = None,
    datasets: Union[list[str], str, None] = None,
) -> list:
    # We use lists from now on for datasets
    if isinstance(datasets, str):
        datasets = [datasets]

    if user is None:
        user = await get_default_user()

    if user is None:
        raise UserNotFoundError

    query = await log_query(query_text, str(query_type), user.id)

    own_document_ids = await get_document_ids_for_user(user.id, datasets)
    search_results = await specific_search(query_type, query_text, user)

    filtered_search_results = []

    for search_result in search_results:
        document_id = search_result["document_id"] if "document_id" in search_result else None
        document_id = UUID(document_id) if isinstance(document_id, str) else document_id

        if document_id is None or document_id in own_document_ids:
            filtered_search_results.append(search_result)

    await log_result(query.id, json.dumps(filtered_search_results, cls=JSONEncoder), user.id)

    return filtered_search_results


async def specific_search(query_type: SearchType, query: str, user) -> list:
    search_tasks: Dict[SearchType, Callable] = {
        SearchType.SUMMARIES: query_summaries,
        SearchType.INSIGHTS: query_graph_connections,
        SearchType.CHUNKS: query_chunks,
        SearchType.COMPLETION: query_completion,
        SearchType.GRAPH_COMPLETION: graph_query_completion,
    }

    search_task = search_tasks.get(query_type)

    if search_task is None:
        raise InvalidValueError(message=f"Unsupported search type: {query_type}")

    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    results = await search_task(query)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return results
