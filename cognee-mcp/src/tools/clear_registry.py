from contextlib import redirect_stdout
import sys
from src.utils.context import cognee_client
from cognee.shared.logging_utils import get_logger
import mcp.types as types

logger = get_logger()


async def clear_registry() -> list:
    """
    Clear all stored MCP server information from the registry.

    Removes all MCP servers you've stored. Use with caution as this cannot be undone.

    Returns
    -------
    list
        A TextContent object confirming the registry was cleared.
    """

    with redirect_stdout(sys.stderr):
        try:
            await cognee_client.prune_data()
            await cognee_client.prune_system(metadata=True)
            logger.info("MCP server registry cleared")
            return [
                types.TextContent(
                    type="text",
                    text="✅ MCP server registry has been cleared. All stored servers removed.",
                )
            ]
        except NotImplementedError:
            error_msg = "❌ Clear operation is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to clear registry: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
