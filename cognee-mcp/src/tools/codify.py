"""Tool for analyzing and generating code-specific knowledge graphs from repositories."""

import sys
import asyncio
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger, get_log_file_location

from src.shared import context

logger = get_logger()


async def codify(repo_path: str) -> list:
    """
    Analyze and generate a code-specific knowledge graph from a software repository.

    This function launches a background task that processes the provided repository
    and builds a code knowledge graph. The function returns immediately while
    the processing continues in the background due to MCP timeout constraints.

    Parameters
    ----------
    repo_path : str
        Path to the code repository to analyze. This can be a local file path or a
        relative path to a repository. The path should point to the root of the
        repository or a specific directory within it.

    Returns
    -------
    list
        A list containing a single TextContent object with information about the
        background task launch and how to check its status.

    Notes
    -----
    - The function launches a background task and returns immediately
    - The code graph generation may take significant time for larger repositories
    - Use the codify_status tool to check the progress of the operation
    - Process results are logged to the standard Cognee log file
    - All stdout is redirected to stderr to maintain MCP communication integrity
    """

    if context.cognee_client.use_api:
        error_msg = "❌ Codify operation is not available in API mode. Please use direct mode for code graph pipeline."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]

    async def codify_task(repo_path: str):
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Codify process starting.")
            from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline

            results = []
            async for result in run_code_graph_pipeline(repo_path, False):
                results.append(result)
                logger.info(result)
            if all(results):
                logger.info("Codify process finished succesfully.")
            else:
                logger.info("Codify process failed.")

    asyncio.create_task(codify_task(repo_path))

    log_file = get_log_file_location()
    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current codify status use the codify_status tool\n"
        f"or you can check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]
