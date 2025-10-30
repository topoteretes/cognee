from contextlib import redirect_stdout
import json
import sys
from cognee.shared.logging_utils import get_logger
import mcp.types as types

from src.utils.context import cognee_client

logger = get_logger()


async def find_mcp_server(requirements: str, max_results: int = 5) -> list:
    """
    Search for MCP servers that match your requirements.

    Searches through stored MCP servers and returns the ones that best match your needs
    based on their capabilities and descriptions.

    Parameters
    ----------
    requirements : str
        Describe what you need the MCP server to do. Be specific about your use case.
        Examples:
        - "I need to read and write files"
        - "I want to search the web for real-time information"
        - "I need to control a browser and take screenshots"
        - "I want to execute code in a sandbox"

    max_results : int, optional
        Maximum number of MCP servers to return (default: 5)

    Returns
    -------
    list
        A TextContent object with detailed information about matching MCP servers,
        including their names, descriptions, capabilities, and installation instructions.

    Examples
    --------
    ```python
    # Find a server for file operations
    await find_mcp_server("I need to read and modify files in my project")

    # Find a server for web search
    await find_mcp_server("I want to search the internet for current information")
    ```
    """

    with redirect_stdout(sys.stderr):
        try:
            logger.info(f"Searching for MCP servers matching: {requirements}")

            # Search using GRAPH_COMPLETION for intelligent matching
            search_results = await cognee_client.search(
                query_text=f"Find MCP servers that can: {requirements}. Include their capabilities, installation instructions, and documentation.",
                query_type="GRAPH_COMPLETION",
            )

            # Format the results
            if cognee_client.use_api:
                if isinstance(search_results, str):
                    result_text = search_results
                elif isinstance(search_results, list) and len(search_results) > 0:
                    result_text = str(search_results[0])
                else:
                    result_text = json.dumps(search_results, cls=json.JSONEncoder)
            else:
                if isinstance(search_results, list) and len(search_results) > 0:
                    result_text = str(search_results[0])
                else:
                    result_text = str(search_results)

            logger.info("MCP server search completed")

            return [
                types.TextContent(
                    type="text",
                    text=f"🔍 MCP Servers matching your requirements:\n\n{result_text}",
                )
            ]

        except Exception as e:
            error_msg = f"❌ Failed to search for MCP servers: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
