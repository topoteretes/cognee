"""Tool for deleting specific data from a dataset."""

import sys
import json
from uuid import UUID
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger
from cognee.modules.storage.utils import JSONEncoder

from src.shared import context

logger = get_logger()


async def delete(data_id: str, dataset_id: str, mode: str = "soft") -> list:
    """
    Delete specific data from a dataset in the Cognee knowledge graph.

    This function removes a specific data item from a dataset while keeping the
    dataset itself intact. It supports both soft and hard deletion modes.

    Parameters
    ----------
    data_id : str
        The UUID of the data item to delete from the knowledge graph.
        This should be a valid UUID string identifying the specific data item.

    dataset_id : str
        The UUID of the dataset containing the data to be deleted.
        This should be a valid UUID string identifying the dataset.

    mode : str, optional
        The deletion mode to use. Options are:
        - "soft" (default): Removes the data but keeps related entities that might be shared
        - "hard": Also removes degree-one entity nodes that become orphaned after deletion
        Default is "soft" for safer deletion that preserves shared knowledge.

    Returns
    -------
    list
        A list containing a single TextContent object with the deletion results,
        including status, deleted node counts, and confirmation details.

    Notes
    -----
    - This operation cannot be undone. The specified data will be permanently removed.
    - Hard mode may remove additional entity nodes that become orphaned
    - The function provides detailed feedback about what was deleted
    - Use this for targeted deletion instead of the prune tool which removes everything
    """

    with redirect_stdout(sys.stderr):
        try:
            logger.info(
                f"Starting delete operation for data_id: {data_id}, dataset_id: {dataset_id}, mode: {mode}"
            )

            # Convert string UUIDs to UUID objects
            data_uuid = UUID(data_id)
            dataset_uuid = UUID(dataset_id)

            # Call the cognee delete function via client
            result = await context.cognee_client.delete(
                data_id=data_uuid, dataset_id=dataset_uuid, mode=mode
            )

            logger.info(f"Delete operation completed successfully: {result}")

            # Format the result for MCP response
            formatted_result = json.dumps(result, indent=2, cls=JSONEncoder)

            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Delete operation completed successfully!\n\n{formatted_result}",
                )
            ]

        except ValueError as e:
            # Handle UUID parsing errors
            error_msg = f"❌ Invalid UUID format: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]

        except Exception as e:
            # Handle all other errors (DocumentNotFoundError, DatasetNotFoundError, etc.)
            error_msg = f"❌ Delete operation failed: {str(e)}"
            logger.error(f"Delete operation error: {str(e)}")
            return [types.TextContent(type="text", text=error_msg)]
