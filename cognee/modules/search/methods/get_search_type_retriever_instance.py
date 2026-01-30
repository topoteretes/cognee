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


async def get_search_type_retriever_instance(
    query_type: SearchType,
    query_text: str,
    **kwargs,
) -> BaseRetriever:
    """
    Factory method to get the appropriate retriever instance based on the search type.

    Args:
        query_type: SearchType enum indicating the type of search.
        query_text: query string.
        retriever_specific_config: Retriever specific configuration dictionary.
        **kwargs: General keyword arguments for retriever initialization.

    Returns:

    """
    # Transform retriever specific config if empty to avoid None checks later
    retriever_specific_config = kwargs.get("retriever_specific_config")
    if retriever_specific_config is None:
        retriever_specific_config = {}

    # Extract common defaults with fallback values from kwargs
    top_k = kwargs.get("top_k", 10)
    system_prompt_path = kwargs.get("system_prompt_path", "answer_simple_question.txt")
    system_prompt = kwargs.get("system_prompt")
    node_type = kwargs.get("node_type", NodeSet)
    node_name = kwargs.get("node_name")
    save_interaction = kwargs.get("save_interaction", False)
    wide_search_top_k = kwargs.get("wide_search_top_k", 100)
    triplet_distance_penalty = kwargs.get("triplet_distance_penalty", 3.5)
    session_id = kwargs.get("session_id")

    # Registry mapping search types to their corresponding retriever classes and input parameters
    search_core_registry: dict[SearchType, Tuple[BaseRetriever, dict]] = {
        SearchType.SUMMARIES: (SummariesRetriever, {"top_k": top_k, "session_id": session_id}),
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
                "session_id": session_id,
                "response_model": retriever_specific_config.get("response_model", str),
            },
        ),
        SearchType.TRIPLET_COMPLETION: (
            TripletRetriever,
            {
                "system_prompt_path": system_prompt_path,
                "top_k": top_k,
                "system_prompt": system_prompt,
                "session_id": session_id,
                "response_model": retriever_specific_config.get("response_model", str),
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
                "session_id": session_id,
                "response_model": retriever_specific_config.get("response_model", str),
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
                "max_iter": retriever_specific_config.get("max_iter", 4),
                "validation_system_prompt_path": retriever_specific_config.get(
                    "validation_system_prompt_path", "cot_validation_system_prompt.txt"
                ),
                "validation_user_prompt_path": retriever_specific_config.get(
                    "validation_user_prompt_path", "cot_validation_user_prompt.txt"
                ),
                "followup_system_prompt_path": retriever_specific_config.get(
                    "followup_system_prompt_path", "cot_followup_system_prompt.txt"
                ),
                "followup_user_prompt_path": retriever_specific_config.get(
                    "followup_user_prompt_path", "cot_followup_user_prompt.txt"
                ),
                "session_id": session_id,
                "response_model": retriever_specific_config.get("response_model", str),
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
                "context_extension_rounds": retriever_specific_config.get(
                    "context_extension_rounds", 4
                ),
                "session_id": session_id,
                "response_model": retriever_specific_config.get("response_model", str),
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
                "session_id": session_id,
                "summarize_prompt_path": retriever_specific_config.get(
                    "summarize_prompt_path", "summarize_search_results.txt"
                ),
            },
        ),
        SearchType.CYPHER: (
            CypherSearchRetriever,
            {
                "user_prompt_path": retriever_specific_config.get(
                    "user_prompt_path", "context_for_question.txt"
                ),
                "system_prompt_path": retriever_specific_config.get(
                    "system_prompt_path", "answer_simple_question.txt"
                ),
                "session_id": session_id,
            },
        ),
        SearchType.NATURAL_LANGUAGE: (
            NaturalLanguageRetriever,
            {
                "session_id": session_id,
                "system_prompt_path": retriever_specific_config.get(
                    "system_prompt_path", "natural_language_retriever_system.txt"
                ),
                "max_attempts": retriever_specific_config.get("max_attempts", 3),
            },
        ),
        SearchType.TEMPORAL: (
            TemporalRetriever,
            {
                "top_k": top_k,
                "wide_search_top_k": wide_search_top_k,
                "triplet_distance_penalty": triplet_distance_penalty,
                "session_id": session_id,
                "response_model": retriever_specific_config.get("response_model", str),
                "user_prompt_path": retriever_specific_config.get(
                    "user_prompt_path", "graph_context_for_question.txt"
                ),
                "system_prompt_path": retriever_specific_config.get(
                    "system_prompt_path", "answer_simple_question.txt"
                ),
                "time_extraction_prompt_path": retriever_specific_config.get(
                    "time_extraction_prompt_path", "extract_query_time.txt"
                ),
                "node_type": node_type,
                "node_name": node_name,
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
        retriever = registered_community_retrievers.get(query_type)

        if not retriever:
            raise UnsupportedSearchTypeError(str(query_type))
        # TODO: Fix community retrievers on the community side so they get all input parameters properly
        retriever_instance = retriever(**kwargs)
    else:
        retriever_info = search_core_registry.get(query_type)
        # Check if retriever info is found for the given query type
        if not retriever_info:
            raise UnsupportedSearchTypeError(str(query_type))

        # If it exists unpack the retriever class and its initialization arguments
        retriever_cls, retriever_args = retriever_info

        retriever_instance = retriever_cls(**retriever_args)

    return retriever_instance
