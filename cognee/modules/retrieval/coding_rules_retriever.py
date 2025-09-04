from cognee.shared.logging_utils import get_logger
from cognee.tasks.codingagents.coding_rule_associations import get_existing_rules

logger = get_logger("CodingRulesRetriever")


class CodingRulesRetriever:
    """Retriever for handling codeing rule based searches."""

    def __init__(self, rules_nodeset_name):
        if isinstance(rules_nodeset_name, list):
            rules_nodeset_name = rules_nodeset_name[0]
        self.rules_nodeset_name = rules_nodeset_name
        """Initialize retriever with search parameters."""

    async def get_existing_rules(self, query_text):
        return await get_existing_rules(
            rules_nodeset_name=self.rules_nodeset_name, return_list=True
        )
