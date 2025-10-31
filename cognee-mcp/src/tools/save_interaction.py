"""Tool for transforming and saving user-agent interactions into structured knowledge."""

import sys
import asyncio
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger, get_log_file_location

from src.shared import context

logger = get_logger()

# Import coding agent rules functions
try:
    from cognee.tasks.codingagents.coding_rule_associations import add_rule_associations
except ModuleNotFoundError:
    from src.codingagents.coding_rule_associations import add_rule_associations


async def save_interaction(data: str) -> list:
    """
    Transform and save a user-agent interaction into structured knowledge.

    Parameters
    ----------
    data : str
        The input string containing user queries and corresponding agent answers.

    Returns
    -------
    list
        A list containing a single TextContent object with information about the background task launch.
    """

    async def save_user_agent_interaction(data: str) -> None:
        """Build knowledge graph from the interaction data"""
        with redirect_stdout(sys.stderr):
            logger.info("Save interaction process starting.")

            await context.cognee_client.add(data, node_set=["user_agent_interaction"])

            try:
                await context.cognee_client.cognify()
                logger.info("Save interaction process finished.")

                # Rule associations only work in direct mode
                if not context.cognee_client.use_api:
                    logger.info("Generating associated rules from interaction data.")
                    await add_rule_associations(data=data, rules_nodeset_name="coding_agent_rules")
                    logger.info("Associated rules generated from interaction data.")
                else:
                    logger.warning("Rule associations are not available in API mode, skipping.")

            except Exception as e:
                logger.error("Save interaction process failed.")
                raise ValueError(f"Failed to Save interaction: {str(e)}")

    asyncio.create_task(
        save_user_agent_interaction(
            data=data,
        )
    )

    log_file = get_log_file_location()
    text = (
        f"Background process launched to process the user-agent interaction.\n"
        f"To check the current status, use the cognify_status tool or check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]
