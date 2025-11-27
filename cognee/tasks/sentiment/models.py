"""Models used by interaction sentiment classification tasks."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import NodeSet
from pydantic import BaseModel, Field, confloat


class InteractionSentimentLabel(str, Enum):
    """Enumerated sentiment labels evaluated for user interactions."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class InteractionSentiment(DataPoint):
    """DataPoint capturing sentiment analysis for a saved Cognee interaction."""

    interaction_id: UUID = Field(..., description="Identifier of the originating interaction")
    sentiment: InteractionSentimentLabel = Field(..., description="Overall sentiment label")
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Model confidence for the classified sentiment in the inclusive range [0, 1]",
    )
    summary: str = Field(..., description="Short natural language summary supporting the label")
    belongs_to_set: Optional[NodeSet] = Field(
        default=None,
        description="Grouping NodeSet for interaction sentiment data points",
    )
    metadata: dict = Field(default_factory=lambda: {"index_fields": []})


class InteractionSentimentEvaluation(BaseModel):
    """Structured response returned by the LLM sentiment classifier."""

    sentiment: InteractionSentimentLabel
    confidence: confloat(ge=0.0, le=1.0)
    summary: str


class InteractionSnapshot(BaseModel):
    """Validated payload describing a Cognee user interaction for sentiment analysis."""

    interaction_id: UUID
    question: str
    answer: str
    context: str = ""
