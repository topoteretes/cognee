import json
from typing import Callable

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine.utils import parse_id
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.insights_retriever import InsightsRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.code_retriever import CodeRetriever
from cognee.modules.search.types import SearchType
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.shared.utils import send_telemetry
from ..operations import log_query, log_result


async def search(
    query_text: str,
    query_type: SearchType,
    datasets: list[str],
    user: User,
):
    query = await log_query(query_text, query_type.value, user.id)

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
        SearchType.SUMMARIES: SummariesRetriever.as_search(),
        SearchType.INSIGHTS: InsightsRetriever.as_search(),
        SearchType.CHUNKS: ChunksRetriever.as_search(),
        SearchType.COMPLETION: CompletionRetriever.as_search(),
        SearchType.GRAPH_COMPLETION: GraphCompletionRetriever.as_search(),
        SearchType.GRAPH_SUMMARY_COMPLETION: GraphSummaryCompletionRetriever.as_search(),
        SearchType.CODE: CodeRetriever.as_search(),
    }

    search_task = search_tasks.get(query_type)

    if search_task is None:
        raise InvalidValueError(message=f"Unsupported search type: {query_type}")

    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    results = await search_task(query)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return results
