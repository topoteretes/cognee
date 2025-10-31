"""Tool for retrieving developer rules from the knowledge graph."""

import sys
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger

from src.shared import context

logger = get_logger()

# Import coding agent rules functions
try:
    from cognee.tasks.codingagents.coding_rule_associations import get_existing_rules
except ModuleNotFoundError:
    from src.codingagents.coding_rule_associations import get_existing_rules


async def get_developer_rules() -> list:
    """
    Retrieve all developer rules that were generated based on previous interactions.

    This tool queries the Cognee knowledge graph and returns a list of developer
    rules.

    Parameters
    ----------
    None

    Returns
    -------
    list
        A list containing a single TextContent object with the retrieved developer rules.
        The format is plain text containing the developer rules in bulletpoints.

    Notes
    -----
    - The specific logic for fetching rules is handled internally.
    - This tool does not accept any parameters and is intended for simple rule inspection use cases.
    """

    async def fetch_rules_from_cognee() -> str:
        """Collect all developer rules from Cognee"""
        with redirect_stdout(sys.stderr):
            if context.cognee_client.use_api:
                logger.warning("Developer rules retrieval is not available in API mode")
                return "Developer rules retrieval is not available in API mode"

            developer_rules = await get_existing_rules(rules_nodeset_name="coding_agent_rules")
            return developer_rules

    rules_text = await fetch_rules_from_cognee()

    return [types.TextContent(type="text", text=rules_text)]
