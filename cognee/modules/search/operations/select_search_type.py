from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("SearchTypeSelector")

GRAPH_NODE_NAME_SEARCH_TYPES = frozenset(
    {
        SearchType.GRAPH_COMPLETION,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
        SearchType.AGENTIC_COMPLETION,
    }
)
NODESET_NODE_NAME_SEARCH_TYPES = GRAPH_NODE_NAME_SEARCH_TYPES | frozenset(
    {
        SearchType.CHUNKS,
        SearchType.RAG_COMPLETION,
        SearchType.HYBRID_COMPLETION,
        SearchType.TEMPORAL,
        SearchType.CODING_RULES,
    }
)
DOCUMENT_NODE_NAME_SEARCH_TYPES = GRAPH_NODE_NAME_SEARCH_TYPES | frozenset(
    {SearchType.CHUNKS, SearchType.RAG_COMPLETION}
)


def supported_search_types_for_node_name(node_type: type | None) -> frozenset[SearchType]:
    if node_type is NodeSet:
        return NODESET_NODE_NAME_SEARCH_TYPES
    if isinstance(node_type, type) and issubclass(node_type, Document):
        return DOCUMENT_NODE_NAME_SEARCH_TYPES
    return GRAPH_NODE_NAME_SEARCH_TYPES


async def select_search_type(
    query: str,
    system_prompt_path: str = "search_type_selector_prompt.txt",
    allowed_search_types: frozenset[SearchType] | None = None,
) -> SearchType:
    """
    Analyzes the query and Selects the best search type.

    Args:
        query: The query to analyze.
        system_prompt_path: The path to the system prompt.

    Returns:
        The best search type given by the LLM.
    """
    default_search_type = (
        SearchType.RAG_COMPLETION
        if allowed_search_types is None or SearchType.RAG_COMPLETION in allowed_search_types
        else SearchType.GRAPH_COMPLETION
    )
    # Search types that cannot answer a free-form natural-language query on
    # their own: CODE runs one exact graph operation driven by a structured
    # code_query, which FEELING_LUCKY has no way to construct.
    excluded_search_types = {SearchType.CODE}
    system_prompt = read_query_prompt(system_prompt_path)

    try:
        response = await LLMGateway.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=str,
        )

        if response.upper() in SearchType.__members__:
            selected_search_type = SearchType(response.upper())
            if selected_search_type in excluded_search_types or (
                allowed_search_types is not None
                and selected_search_type not in allowed_search_types
            ):
                logger.info(
                    f"LLM selected {selected_search_type.value}, which is not allowed for this "
                    f"search; falling back to {default_search_type.value}."
                )
                return default_search_type
            logger.info(f"Selected lucky search type: {response.upper()}")
            return selected_search_type

        # If the response is not a valid search type, return the default search type
        logger.info(f"LLM gives an invalid search type: {response.upper()}")
        return default_search_type
    except Exception:
        logger.exception("Failed to select search type intelligently from LLM")
        return default_search_type
