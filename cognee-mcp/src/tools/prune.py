"""Tool for resetting the Cognee knowledge graph."""

import sys
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger

from src.shared import context

logger = get_logger()


async def prune():
    """
    Reset the Cognee knowledge graph by removing all stored information.

    This function performs a complete reset of both the data layer and system layer
    of the Cognee knowledge graph, removing all nodes, edges, and associated metadata.
    It is typically used during development or when needing to start fresh with a new
    knowledge base.

    Returns
    -------
    list
        A list containing a single TextContent object with confirmation of the prune operation.

    Notes
    -----
    - This operation cannot be undone. All memory data will be permanently deleted.
    - The function prunes both data content (using prune_data) and system metadata (using prune_system)
    - This operation is not available in API mode
    """
    with redirect_stdout(sys.stderr):
        try:
            await context.cognee_client.prune_data()
            await context.cognee_client.prune_system(metadata=True)
            return [types.TextContent(type="text", text="Pruned")]
        except NotImplementedError:
            error_msg = "❌ Prune operation is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Prune operation failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
