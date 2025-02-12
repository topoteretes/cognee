import json
from typing import Callable

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine.utils import parse_id
from cognee.modules.retrieval.code_graph_retrieval import code_graph_retrieval
from cognee.modules.search.types import SearchType
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.shared.utils import send_telemetry
from cognee.tasks.chunks import query_chunks
from cognee.tasks.graph import query_graph_connections
from cognee.tasks.summarization import query_summaries
from cognee.tasks.completion import query_completion
from cognee.tasks.completion import graph_query_completion
from ..operations import log_query, log_result


async def search(
    query_text: str,
    query_type: str,
    datasets: list[str],
    user: User,
):
    query = await log_query(query_text, str(query_type), user.id)

    own_document_ids = await get_document_ids_for_user(user.id, datasets)
    search_results = await specific_search(query_type, query_text, user)

    filtered_search_results = []

    for search_result in search_results:
        document_id = search_result["document_id"] if "document_id" in search_result else None
        document_id = parse_id(document_id)

        if document_id is None or document_id in own_document_ids:
            filtered_search_results.append(search_result)

    await log_result(query.id, json.dumps(filtered_search_results, cls=JSONEncoder), user.id)

    return filtered_search_results


async def specific_search(query_type: SearchType, query: str, user: User) -> list:
    search_tasks: dict[SearchType, Callable] = {
        SearchType.SUMMARIES: query_summaries,
        SearchType.INSIGHTS: query_graph_connections,
        SearchType.CHUNKS: query_chunks,
        SearchType.COMPLETION: query_completion,
        SearchType.GRAPH_COMPLETION: graph_query_completion,
        SearchType.CODE: code_graph_retrieval,
    }

    search_task = search_tasks.get(query_type)

    if search_task is None:
        raise InvalidValueError(message=f"Unsupported search type: {query_type}")

    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    results = await search_task(query)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return results
