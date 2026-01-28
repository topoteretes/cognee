import asyncio
from functools import reduce
from typing import List, Optional, Any
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.tasks.codingagents.coding_rule_associations import get_existing_rules

logger = get_logger("CodingRulesRetriever")


class CodingRulesRetriever(BaseRetriever):
    """Retriever for handling codeing rule based searches."""

    def __init__(self, rules_nodeset_name: Optional[List[str]] = None):
        if isinstance(rules_nodeset_name, list) or rules_nodeset_name is None:
            if not rules_nodeset_name:
                # If there is no provided nodeset set to coding_agent_rules
                rules_nodeset_name = ["coding_agent_rules"]

        self.rules_nodeset_name = rules_nodeset_name
        """Initialize retriever with search parameters."""

    async def get_retrieved_objects(self, query: str) -> Any:
        if self.rules_nodeset_name:
            rules_list = await asyncio.gather(
                *[
                    get_existing_rules(rules_nodeset_name=nodeset)
                    for nodeset in self.rules_nodeset_name
                ]
            )
            return reduce(lambda x, y: x + y, rules_list, [])

    async def get_context_from_objects(self, query, retrieved_objects):
        return retrieved_objects

    async def get_completion_from_context(self, query, retrieved_objects, context):
        # TODO: Add completion generation logic if needed
        return context
