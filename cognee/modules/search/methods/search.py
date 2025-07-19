import os
import json
import asyncio
from uuid import UUID
from typing import Callable, List, Optional, Type, Union

from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import InvalidValueError
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.insights_retriever import InsightsRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.code_retriever import CodeRetriever
from cognee.modules.retrieval.cypher_search_retriever import CypherSearchRetriever
from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever
from cognee.modules.search.types import SearchType
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.shared.utils import send_telemetry
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from cognee.modules.search.operations import log_query, log_result


async def search(
    query_text: str,
    query_type: SearchType,
    dataset_ids: Union[list[UUID], None],
    user: User,
    system_prompt_path="answer_simple_question.txt",
    top_k: int = 10,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
):
    """

    Args:
        query_text:
        query_type:
        datasets:
        user:
        system_prompt_path:
        top_k:

    Returns:

    Notes:
        Searching by dataset is only available in ENABLE_BACKEND_ACCESS_CONTROL mode
    """
    # Use search function filtered by permissions if access control is enabled
    if os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true":
        return await authorized_search(
            query_text, query_type, user, dataset_ids, system_prompt_path, top_k
        )

    query = await log_query(query_text, query_type.value, user.id)

    search_results = await specific_search(
        query_type,
        query_text,
        user,
        system_prompt_path=system_prompt_path,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
    )

    await log_result(
        query.id,
        json.dumps(
            search_results if len(search_results) > 1 else search_results[0], cls=JSONEncoder
        ),
        user.id,
    )

    return search_results


async def specific_search(
    query_type: SearchType,
    query: str,
    user: User,
    system_prompt_path="answer_simple_question.txt",
    top_k: int = 10,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
) -> list:
    search_tasks: dict[SearchType, Callable] = {
        SearchType.SUMMARIES: SummariesRetriever(top_k=top_k).get_completion,
        SearchType.INSIGHTS: InsightsRetriever(top_k=top_k).get_completion,
        SearchType.CHUNKS: ChunksRetriever(top_k=top_k).get_completion,
        SearchType.RAG_COMPLETION: CompletionRetriever(
            system_prompt_path=system_prompt_path, top_k=top_k
        ).get_completion,
        SearchType.GRAPH_COMPLETION: GraphCompletionRetriever(
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
        ).get_completion,
        SearchType.GRAPH_COMPLETION_COT: GraphCompletionCotRetriever(
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
        ).get_completion,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: GraphCompletionContextExtensionRetriever(
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
        ).get_completion,
        SearchType.GRAPH_SUMMARY_COMPLETION: GraphSummaryCompletionRetriever(
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
        ).get_completion,
        SearchType.CODE: CodeRetriever(top_k=top_k).get_completion,
        SearchType.CYPHER: CypherSearchRetriever().get_completion,
        SearchType.NATURAL_LANGUAGE: NaturalLanguageRetriever().get_completion,
    }

    search_task = search_tasks.get(query_type)

    if search_task is None:
        raise InvalidValueError(message=f"Unsupported search type: {query_type}")

    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    results = await search_task(query)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return results


async def authorized_search(
    query_text: str,
    query_type: SearchType,
    user: User = None,
    dataset_ids: Optional[list[UUID]] = None,
    system_prompt_path: str = "answer_simple_question.txt",
    top_k: int = 10,
) -> list:
    """
    Verifies access for provided datasets or uses all datasets user has read access for and performs search per dataset.
    Not to be used outside of active access control mode.
    """

    query = await log_query(query_text, query_type.value, user.id)

    # Find datasets user has read access for (if datasets are provided only return them. Provided user has read access)
    search_datasets = await get_specific_user_permission_datasets(user.id, "read", dataset_ids)

    # Searches all provided datasets and handles setting up of appropriate database context based on permissions
    search_results = await specific_search_by_context(
        search_datasets, query_text, query_type, user, system_prompt_path, top_k
    )

    await log_result(query.id, json.dumps(search_results, cls=JSONEncoder), user.id)

    return search_results


async def specific_search_by_context(
    search_datasets: list[Dataset],
    query_text: str,
    query_type: SearchType,
    user: User,
    system_prompt_path: str,
    top_k: int,
):
    """
    Searches all provided datasets and handles setting up of appropriate database context based on permissions.
    Not to be used outside of active access control mode.
    """

    async def _search_by_context(dataset, user, query_type, query_text, system_prompt_path, top_k):
        # Set database configuration in async context for each dataset user has access for
        await set_database_global_context_variables(dataset.id, dataset.owner_id)
        search_results = await specific_search(
            query_type, query_text, user, system_prompt_path=system_prompt_path, top_k=top_k
        )
        return {
            "search_result": search_results,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
        }

    # Search every dataset async based on query and appropriate database configuration
    tasks = []
    for dataset in search_datasets:
        tasks.append(
            _search_by_context(dataset, user, query_type, query_text, system_prompt_path, top_k)
        )

    return await asyncio.gather(*tasks)
