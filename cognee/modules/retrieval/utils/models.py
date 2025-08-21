from typing import Optional
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from enum import Enum
from pydantic import BaseModel, Field, confloat


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
    score: float
    belongs_to_set: Optional[NodeSet] = None


class UserFeedbackSentiment(str, Enum):
    """User - User feedback sentiment"""

    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class UserFeedbackEvaluation(BaseModel):
    """User - User feedback evaluation"""

    score: confloat(ge=-5, le=5) = Field(
        ..., description="Sentiment score from -5 (negative) to +5 (positive)"
    )
    evaluation: UserFeedbackSentiment
