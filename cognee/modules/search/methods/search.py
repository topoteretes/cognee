import os
import json
import asyncio
from uuid import UUID
from functools import reduce
from fastapi.encoders import jsonable_encoder
from typing import Callable, List, Optional, Tuple, Type, Union

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.user_qa_feedback import UserQAFeedback
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.search.exceptions import UnsupportedSearchTypeError
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.insights_retriever import InsightsRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.modules.retrieval.coding_rules_retriever import CodingRulesRetriever
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
from cognee.modules.search.operations import log_query, log_result, select_search_type
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.shared.utils import send_telemetry
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets


async def search(
    query_text: str,
    query_type: SearchType,
    dataset_ids: Union[list[UUID], None],
    user: User,
    system_prompt_path="answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: Optional[bool] = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    use_combined_context: bool = True,
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
    query = await log_query(query_text, query_type.value, user.id)

    # Use search function filtered by permissions if access control is enabled
    if os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true":
        search_results = await authorized_search(
            query_type=query_type,
            query_text=query_text,
            user=user,
            dataset_ids=dataset_ids,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            last_k=last_k,
            only_context=only_context,
            use_combined_context=use_combined_context,
        )
    else:
        search_results = await specific_search(
            query_type=query_type,
            query_text=query_text,
            user=user,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            last_k=last_k,
            only_context=only_context,
        )

    await log_result(
        query.id,
        json.dumps(jsonable_encoder(search_results)),
        user.id,
    )

    return search_results


async def specific_search(
    query_type: SearchType,
    query_text: str,
    user: User,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: Optional[bool] = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    context: Optional[str] = None,
) -> list:
    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    search_tasks: dict[SearchType, List[Callable]] = {
        SearchType.SUMMARIES: [
            SummariesRetriever(top_k=top_k).get_completion,
            SummariesRetriever(top_k=top_k).get_context,
        ],
        SearchType.INSIGHTS: [
            InsightsRetriever(top_k=top_k).get_completion,
            InsightsRetriever(top_k=top_k).get_context,
        ],
        SearchType.CHUNKS: [
            ChunksRetriever(top_k=top_k).get_completion,
            ChunksRetriever(top_k=top_k).get_context,
        ],
        SearchType.RAG_COMPLETION: [
            CompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                system_prompt=system_prompt,
            ).get_completion,
            CompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.GRAPH_COMPLETION: [
            GraphCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_completion,
            GraphCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.GRAPH_COMPLETION_COT: [
            GraphCompletionCotRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_completion,
            GraphCompletionCotRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: [
            GraphCompletionContextExtensionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_completion,
            GraphCompletionContextExtensionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.GRAPH_SUMMARY_COMPLETION: [
            GraphSummaryCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_completion,
            GraphSummaryCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.CODE: [
            CodeRetriever(top_k=top_k).get_completion,
            CodeRetriever(top_k=top_k).get_context,
        ],
        SearchType.CYPHER: [
            CypherSearchRetriever().get_completion,
            CypherSearchRetriever().get_context,
        ],
        SearchType.NATURAL_LANGUAGE: [
            NaturalLanguageRetriever().get_completion,
            NaturalLanguageRetriever().get_context,
        ],
        SearchType.FEEDBACK: [UserQAFeedback(last_k=last_k).add_feedback],
        SearchType.TEMPORAL: [
            TemporalRetriever(top_k=top_k).get_completion,
            TemporalRetriever(top_k=top_k).get_context,
        ],
        SearchType.CODING_RULES: [
            CodingRulesRetriever(rules_nodeset_name=node_name).get_existing_rules,
        ],
    }

    # If the query type is FEELING_LUCKY, select the search type intelligently
    if query_type is SearchType.FEELING_LUCKY:
        query_type = await select_search_type(query_text)

    search_type_tools = search_tasks.get(query_type)

    if not search_type_tools:
        raise UnsupportedSearchTypeError(str(query_type))

    [get_completion, get_context] = search_type_tools

    if only_context:
        return await get_context(query_text)

    results = await get_completion(query_text, context)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return results


async def authorized_search(
    query_type: SearchType,
    query_text: str,
    user: User,
    dataset_ids: Optional[list[UUID]] = None,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: Optional[bool] = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    use_combined_context: bool = False,
) -> Union[List, Tuple[str, List]]:
    """
    Verifies access for provided datasets or uses all datasets user has read access for and performs search per dataset.
    Not to be used outside of active access control mode.
    """
    # Find datasets user has read access for (if datasets are provided only return them. Provided user has read access)
    search_datasets = await get_specific_user_permission_datasets(user.id, "read", dataset_ids)

    if use_combined_context:
        search_responses = await specific_search_by_context(
            search_datasets=search_datasets,
            query_type=query_type,
            query_text=query_text,
            user=user,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            last_k=last_k,
            only_context=True,
        )

        graph = {}
        context = {}
        datasets = []

        for search_response in search_responses:
            dataset_id = str(search_response["dataset_id"])
            graph[dataset_id] = search_response["graph"]
            context[dataset_id] = search_response["context"]
            datasets.append(
                {
                    "id": search_response["dataset_id"],
                    "name": search_response["dataset_name"],
                }
            )

        completion = await generate_completion(
            query=query_text,
            context="\n".join(list(context.values())),
            user_prompt_path="context_for_question.txt",
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
        )

        return [
            {
                "search_result": [completion],
                "graph": reduce(
                    lambda whole_graph, graph_values: whole_graph.extend(graph_values),
                    list(graph.values()),
                ),
                "context": context,
                "dataset_id": [dataset["id"] for dataset in datasets],
                "dataset_name": [dataset["name"] for dataset in datasets],
            }
        ]

    # Searches all provided datasets and handles setting up of appropriate database context based on permissions
    search_results = await specific_search_by_context(
        search_datasets=search_datasets,
        query_type=query_type,
        query_text=query_text,
        user=user,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
        save_interaction=save_interaction,
        last_k=last_k,
        only_context=only_context,
    )

    return search_results


async def specific_search_by_context(
    search_datasets: list[Dataset],
    query_type: SearchType,
    query_text: str,
    user: User,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: Optional[bool] = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    context: Optional[str] = None,
):
    """
    Searches all provided datasets and handles setting up of appropriate database context based on permissions.
    Not to be used outside of active access control mode.
    """

    async def _search_by_context(
        dataset: Dataset,
        query_type: SearchType,
        query_text: str,
        user: User,
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: int = 10,
        node_type: Optional[Type] = NodeSet,
        node_name: Optional[List[str]] = None,
        save_interaction: Optional[bool] = False,
        last_k: Optional[int] = None,
        only_context: bool = False,
        context: Optional[str] = None,
    ):
        # Set database configuration in async context for each dataset user has access for
        await set_database_global_context_variables(dataset.id, dataset.owner_id)

        search_result_or_context = await specific_search(
            query_type=query_type,
            query_text=query_text,
            user=user,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            last_k=last_k,
            only_context=only_context,
            context=context,
        )

        if only_context and isinstance(
            search_result_or_context, tuple
        ):  # In some cases when only the context is requested, we get (context, triplets)
            search_results = ""
            context = search_result_or_context[0]
            triplets = search_result_or_context[1]
        else:
            search_results = search_result_or_context
            context = []
            triplets = []

        return {
            "search_result": search_results,
            "context": context,
            "graph": [
                {
                    "source": {
                        "id": triplet.node1.id,
                        "attributes": {
                            "name": triplet.node1.attributes["name"],
                            "type": triplet.node1.attributes["type"],
                            "description": triplet.node1.attributes["description"],
                            "vector_distance": triplet.node1.attributes["vector_distance"],
                        },
                    },
                    "destination": {
                        "id": triplet.node2.id,
                        "attributes": {
                            "name": triplet.node2.attributes["name"],
                            "type": triplet.node2.attributes["type"],
                            "description": triplet.node2.attributes["description"],
                            "vector_distance": triplet.node2.attributes["vector_distance"],
                        },
                    },
                    "attributes": {
                        "relationship_name": triplet.attributes["relationship_name"],
                    },
                }
                for triplet in triplets
            ],
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
        }

    # Search every dataset async based on query and appropriate database configuration
    tasks = []
    for dataset in search_datasets:
        tasks.append(
            _search_by_context(
                dataset=dataset,
                query_type=query_type,
                query_text=query_text,
                user=user,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                last_k=last_k,
                only_context=only_context,
                context=context,
            )
        )

    return await asyncio.gather(*tasks)
