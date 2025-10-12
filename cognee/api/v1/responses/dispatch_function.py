import json
import logging
from typing import Any, Dict, Union

from cognee.api.v1.responses.models import ToolCall
from cognee.modules.search.types import SearchType
from cognee.api.v1.add import add
from cognee.api.v1.search import search
from cognee.api.v1.cognify import cognify
from cognee.api.v1.prune import prune


from cognee.modules.users.methods import get_default_user
from cognee.api.v1.responses.default_tools import DEFAULT_TOOLS

logger = logging.getLogger(__name__)


async def dispatch_function(tool_call: Union[ToolCall, Dict[str, Any]]) -> str:
    """
    Dispatches a function call to the appropriate Cognee function.
    """
    if isinstance(tool_call, dict):
        function_data = tool_call.get("function", {})
        function_name = function_data.get("name", "")
        arguments_str = function_data.get("arguments", "{}")
    else:
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments

    arguments = json.loads(arguments_str)

    logger.info(f"Dispatching function: {function_name} with args: {arguments}")

    user = await get_default_user()

    if function_name == "search":
        return await handle_search(arguments, user)
    elif function_name == "cognify":
        return await handle_cognify(arguments, user)
    elif function_name == "prune":
        return await handle_prune(arguments, user)
    else:
        return f"Error: Unknown function {function_name}"


async def handle_search(arguments: Dict[str, Any], user) -> list:
    """Handle search function call"""
    search_tool = next((tool for tool in DEFAULT_TOOLS if tool["name"] == "search"), None)
    required_params = (
        search_tool["parameters"].get("required", []) if search_tool else ["search_query"]
    )

    query = arguments.get("search_query")
    if not query and "search_query" in required_params:
        return "Error: Missing required 'search_query' parameter"

    search_type_str = arguments.get("search_type", "GRAPH_COMPLETION")
    valid_search_types = (
        search_tool["parameters"]["properties"]["search_type"]["enum"]
        if search_tool
        else ["CODE", "GRAPH_COMPLETION", "NATURAL_LANGUAGE"]
    )

    if search_type_str not in valid_search_types:
        logger.warning(f"Invalid search_type: {search_type_str}, defaulting to GRAPH_COMPLETION")
        search_type_str = "GRAPH_COMPLETION"

    query_type = SearchType[search_type_str]

    top_k = arguments.get("top_k")
    datasets = arguments.get("datasets")
    system_prompt_path = arguments.get("system_prompt_path", "answer_simple_question.txt")

    results = await search(
        query_text=query,
        query_type=query_type,
        datasets=datasets,
        user=user,
        system_prompt_path=system_prompt_path,
        top_k=top_k if isinstance(top_k, int) else 10,
    )

    return results


async def handle_cognify(arguments: Dict[str, Any], user) -> str:
    """Handle cognify function call"""
    text = arguments.get("text")
    ontology_file_path = arguments.get("ontology_file_path")
    custom_prompt = arguments.get("custom_prompt")

    if text:
        await add(data=text, user=user)

    await cognify(
        user=user,
        ontology_file_path=ontology_file_path if ontology_file_path else None,
        custom_prompt=custom_prompt,
    )

    return (
        "Text successfully converted into knowledge graph."
        if text
        else "Knowledge graph successfully updated with new information."
    )


async def handle_prune(arguments: Dict[str, Any], user) -> str:
    """Handle prune function call"""
    await prune()
    return "Memory has been pruned successfully."
