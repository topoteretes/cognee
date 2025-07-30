from typing import Optional
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.engine.models.node_set import NodeSet


class CogneeUserInteraction(DataPoint):
    """User - Cognee interaction"""

    question: str
    answer: str
    context: str
    belongs_to_set: Optional[NodeSet] = None


class CogneeUserFeedback(DataPoint):
    """User - Cognee Feedback"""

    feedback: str
    belongs_to_set: Optional[NodeSet] = None
