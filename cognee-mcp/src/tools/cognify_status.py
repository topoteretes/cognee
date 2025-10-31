"""Tool for getting the status of the cognify pipeline."""

import sys
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger

from src.shared import context

logger = get_logger()


async def cognify_status():
    """
    Get the current status of the cognify pipeline.

    This function retrieves information about current and recently completed cognify operations
    in the main_dataset. It provides details on progress, success/failure status, and statistics
    about the processed data.

    Returns
    -------
    list
        A list containing a single TextContent object with the status information as a string.
        The status includes information about active and completed jobs for the cognify_pipeline.

    Notes
    -----
    - The function retrieves pipeline status specifically for the "cognify_pipeline" on the "main_dataset"
    - Status information includes job progress, execution time, and completion status
    - The status is returned in string format for easy reading
    - This operation is not available in API mode
    """
    with redirect_stdout(sys.stderr):
        try:
            from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
            from cognee.modules.users.methods import get_default_user

            user = await get_default_user()
            status = await context.cognee_client.get_pipeline_status(
                [await get_unique_dataset_id("main_dataset", user)], "cognify_pipeline"
            )
            return [types.TextContent(type="text", text=str(status))]
        except NotImplementedError:
            error_msg = "❌ Pipeline status is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to get cognify status: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
