from typing import List
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


class QueryState:
    """
    Helper class containing all necessary information about the query state.
    Used (for now) in COT and Context Extension Retrievers to keep track of important information
    in a more readable way, and enable as many parallel calls to llms as possible.
    """

    def __init__(
        self,
        triplets: List[Edge] = None,
        context_text: str = "",
        finished_extending_context: bool = False,
    ):
        # Mutual fields for COT and Context Extension
        self.triplets = triplets if triplets else []
        self.context_text = context_text
        self.completion = ""

        # Context Extension specific
        self.finished_extending_context = finished_extending_context

        # COT specific
        self.answer_text: str = ""
        self.valid_user_prompt: str = ""
        self.valid_system_prompt: str = ""
        self.reasoning: str = ""

        self.followup_question: str = ""
        self.followup_prompt: str = ""
        self.followup_system: str = ""
