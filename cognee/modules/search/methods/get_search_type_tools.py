import os
from typing import Callable, List, Optional, Type, Tuple

from cognee.modules.retrieval.base_retriever import BaseRetriever

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType
from cognee.modules.search.operations import select_search_type
from cognee.modules.search.exceptions import UnsupportedSearchTypeError

# Retrievers
from cognee.modules.retrieval.user_qa_feedback import UserQAFeedback
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.modules.retrieval.coding_rules_retriever import CodingRulesRetriever
from cognee.modules.retrieval.jaccard_retrival import JaccardChunksRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.cypher_search_retriever import CypherSearchRetriever
from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever


async def get_search_type_tools(
    query_type: SearchType,
    query_text: str,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    last_k: Optional[int] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> list:
    # Registry mapping search types to their corresponding retriever classes and input parameters
    search_core_registry: dict[SearchType, Tuple[BaseRetriever, dict]] = {
        SearchType.SUMMARIES: (SummariesRetriever, {"top_k": top_k}),
        SearchType.CHUNKS: (
            ChunksRetriever,
            {"top_k": top_k},
        ),
        SearchType.RAG_COMPLETION: (
            CompletionRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "system_prompt": system_prompt,
            },
        ),
        SearchType.TRIPLET_COMPLETION: (
            TripletRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "system_prompt": system_prompt,
            },
        ),
        SearchType.GRAPH_COMPLETION: (
            GraphCompletionRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "node_type": node_type,
                "node_name": node_name,
                "save_interaction": save_interaction,
                "system_prompt": system_prompt,
                "wide_search_top_k": wide_search_top_k,
                "triplet_distance_penalty": triplet_distance_penalty,
            },
        ),
        SearchType.GRAPH_COMPLETION_COT: (
            GraphCompletionCotRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "node_type": node_type,
                "node_name": node_name,
                "save_interaction": save_interaction,
                "system_prompt": system_prompt,
                "wide_search_top_k": wide_search_top_k,
                "triplet_distance_penalty": triplet_distance_penalty,
            },
        ),
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: (
            GraphCompletionContextExtensionRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "node_type": node_type,
                "node_name": node_name,
                "save_interaction": save_interaction,
                "system_prompt": system_prompt,
                "wide_search_top_k": wide_search_top_k,
                "triplet_distance_penalty": triplet_distance_penalty,
            },
        ),
        SearchType.GRAPH_SUMMARY_COMPLETION: (
            GraphSummaryCompletionRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "node_type": node_type,
                "node_name": node_name,
                "save_interaction": save_interaction,
                "system_prompt": system_prompt,
                "wide_search_top_k": wide_search_top_k,
                "triplet_distance_penalty": triplet_distance_penalty,
            },
        ),
        SearchType.CYPHER: (CypherSearchRetriever, {}),
        SearchType.NATURAL_LANGUAGE: (NaturalLanguageRetriever, {}),
        # TODO: Remove UserQAFeedback retriever once feedback mechanism is revamped
        SearchType.FEEDBACK: (UserQAFeedback(last_k=last_k)).add_feedback,
        SearchType.TEMPORAL: (
            TemporalRetriever,
            {
                "top_k": top_k,
                "wide_search_top_k": wide_search_top_k,
                "triplet_distance_penalty": triplet_distance_penalty,
            },
        ),
        SearchType.CHUNKS_LEXICAL: (JaccardChunksRetriever, {"top_k": top_k}),
        SearchType.CODING_RULES: (
            CodingRulesRetriever,
            {"rules_nodeset_name": node_name},
        ),
    }

    # If the query type is FEELING_LUCKY, select the search type intelligently
    if query_type is SearchType.FEELING_LUCKY:
        query_type = await select_search_type(query_text)

    if (
        query_type in [SearchType.CYPHER, SearchType.NATURAL_LANGUAGE]
        and os.getenv("ALLOW_CYPHER_QUERY", "true").lower() == "false"
    ):
        raise UnsupportedSearchTypeError("Cypher query search types are disabled.")

    from cognee.modules.retrieval.registered_community_retrievers import (
        registered_community_retrievers,
    )

    if query_type in registered_community_retrievers:
        retriever = registered_community_retrievers[query_type]
        retriever_instance = retriever(top_k=top_k)
        search_type_tools = [
            retriever_instance.get_completion,
            retriever_instance.get_context,
        ]
    else:
        retriever_cls, kwargs = search_core_registry.get(query_type)
        retriever_instance = retriever_cls(**kwargs)
        search_type_tools = [
            retriever_instance.get_completion,
            retriever_instance.get_context,
        ]

    if not search_type_tools:
        raise UnsupportedSearchTypeError(str(query_type))

    return search_type_tools
