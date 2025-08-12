from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.LLMGateway import LLMGateway

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
    system_prompt = read_query_prompt(system_prompt_path)

    try:
        response = await LLMGateway.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=str,
        )

        if response.upper() in SearchType.__members__:
            logger.info(f"Selected lucky search type: {response.upper()}")
            return SearchType(response.upper())

        # If the response is not a valid search type, return the default search type
        logger.info(f"LLM gives an invalid search type: {response.upper()}")
        return default_search_type
    except Exception as e:
        logger.error(f"Failed to select search type intelligently from LLM: {str(e)}")
        return default_search_type
