from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("SearchTypeSelector")


async def select_search_type(
    query: str,
    system_prompt_path: str = "search_type_selector_prompt.txt",
) -> SearchType:
    """
    Analyzes the query and Selects the best search type.

    Args:
        query: The query to analyze.
        system_prompt_path: The path to the system prompt.

    Returns:
        The best search type given by the LLM.
    """
    default_search_type = SearchType.RAG_COMPLETION
    # Search types FEELING_LUCKY must never route to. CODE runs one exact graph
    # operation driven by a structured code_query, which FEELING_LUCKY has no way
    # to construct. The Cypher-executing types need write permission on every
    # dataset searched (SearchType.required_permissions), and the datasets were
    # resolved with read only before this selector runs.
    excluded_search_types = {SearchType.CODE} | {
        search_type for search_type in SearchType if "write" in search_type.required_permissions
    }
    system_prompt = read_query_prompt(system_prompt_path)

    try:
        response = await LLMGateway.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=str,
        )

        if response.upper() in SearchType.__members__:
            selected_search_type = SearchType(response.upper())
            if selected_search_type in excluded_search_types:
                logger.info(
                    f"LLM selected {selected_search_type.value}, which FEELING_LUCKY cannot "
                    f"route to; falling back to {default_search_type.value}."
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
