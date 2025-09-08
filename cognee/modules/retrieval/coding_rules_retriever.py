import asyncio
from functools import reduce
from typing import List, Optional
from cognee.shared.logging_utils import get_logger
from cognee.tasks.codingagents.coding_rule_associations import get_existing_rules

logger = get_logger("CodingRulesRetriever")


class CodingRulesRetriever:
    """Retriever for handling codeing rule based searches."""

    def __init__(self, rules_nodeset_name: Optional[List[str]] = None):
        if isinstance(rules_nodeset_name, list):
            if not rules_nodeset_name:
                # If there is no provided nodeset set to coding_agent_rules
                rules_nodeset_name = ["coding_agent_rules"]

        self.rules_nodeset_name = rules_nodeset_name
        """Initialize retriever with search parameters."""

    async def get_existing_rules(self, query_text):
        if self.rules_nodeset_name:
            rules_list = await asyncio.gather(
                *[
                    get_existing_rules(rules_nodeset_name=nodeset)
                    for nodeset in self.rules_nodeset_name
                ]
            )

            return reduce(lambda x, y: x + y, rules_list, [])
