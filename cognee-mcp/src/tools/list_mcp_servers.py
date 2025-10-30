from contextlib import redirect_stdout
import json
import sys
from cognee.shared.logging_utils import get_logger
import mcp.types as types

from src.utils.context import cognee_client

logger = get_logger()


async def list_mcp_servers() -> list:
    """
    List all MCP servers stored in your personal registry.

    Returns detailed information about MCP servers you've previously remembered,
    including their connection details (URL/command), capabilities, and documentation.
    Use this information to connect to the servers with your MCP client.

    Returns
    -------
    list
        A list of all MCP servers in the registry with their connection information.
    """

    with redirect_stdout(sys.stderr):
        try:
            logger.info("Listing all MCP servers")

            # Search for all MCP servers with connection details
            search_results = await cognee_client.search(
                query_text="List all MCP servers with their names, descriptions, capabilities, connection information (URL, command, args), installation instructions, and documentation links",
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

            output_text = f"📋 MCP Servers in Registry:\n\n{result_text}\n\n"
            output_text += "💡 Use the connection information above (URL or command/args) to configure your MCP client."

            logger.info("MCP server listing completed")

            return [
                types.TextContent(
                    type="text",
                    text=output_text,
                )
            ]

        except Exception as e:
            error_msg = f"❌ Failed to list MCP servers: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
