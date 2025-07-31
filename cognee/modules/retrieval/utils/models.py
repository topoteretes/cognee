from typing import Optional
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from enum import Enum
from pydantic import BaseModel, ValidationError


class CogneeUserInteraction(DataPoint):
    """User - Cognee interaction"""

    question: str
    answer: str
    context: str
    belongs_to_set: Optional[NodeSet] = None


class CogneeUserFeedback(DataPoint):
    """User - Cognee Feedback"""

    feedback: str
    sentiment: str
    belongs_to_set: Optional[NodeSet] = None


class UserFeedbackSentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class UserFeedbackEvaluation(BaseModel):
    evaluation: UserFeedbackSentiment
