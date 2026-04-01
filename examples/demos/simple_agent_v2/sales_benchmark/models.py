"""Pydantic output models for the sales benchmark."""

from __future__ import annotations

from typing import Literal, Optional
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import NodeSet


class SalesResponse(BaseModel):
    """Sales agent turn output."""

    thought: str = Field(min_length=1, description="Internal reasoning about strategy for this lead")
    lead_feature: str = Field(
        min_length=1,
        description=(
            "Which Cognee feature to lead with: multimodal_ingestion, knowledge_structuring, "
            "access_control, retrieval, memory, or feedback"
        ),
    )
    pitch_angle: str = Field(
        min_length=1,
        description="Framing: developer_experience, compliance, roi, simplicity, or research",
    )
    message_to_customer: str = Field(
        min_length=1, max_length=500, description="The pitch message the customer will see"
    )


class CustomerHint(BaseModel):
    """Customer LLM generates natural response text; outcome is deterministic."""

    message: str = Field(
        min_length=1, max_length=500, description="Natural language response to the sales agent"
    )


class ConversationResult(BaseModel):
    """Final result of one lead conversation."""

    lead_id: str
    persona_tag: str
    outcome: Literal["CLOSED_WON", "CLOSED_LOST"]
    rounds: int
    features_pitched: list[str]
    winning_feature: Optional[str] = None
    winning_angle: Optional[str] = None
    conversation_history: list[dict] = []


class ContextSummary(BaseModel):
    """LLM-generated summary of past conversation transcripts."""

    summary: str = Field(
        min_length=1,
        description=(
            "A concise summary of all past sales conversations. "
            "For each persona type, state which features and pitch angles led to "
            "CLOSED_WON vs CLOSED_LOST outcomes."
        ),
    )


# ---------------------------------------------------------------------------
# Graph DataPoint models for structured memory
# ---------------------------------------------------------------------------
# These create shared nodes in the graph so that traces with the same feature,
# angle, outcome, or problem type are connected via graph edges.
# Deterministic IDs (uuid5) ensure the same entity always maps to the same node.


def _deterministic_id(namespace: str, name: str):
    return uuid5(NAMESPACE_OID, f"{namespace}:{name}")


class SalesFeatureNode(DataPoint):
    """A Cognee feature (shared node — all traces pitching this feature link here)."""

    id: object = None  # set via factory
    name: str
    metadata: dict = {"index_fields": ["name"]}

    def __init__(self, name: str, **kwargs):
        super().__init__(id=_deterministic_id("SalesFeature", name), name=name, **kwargs)


class PitchAngleNode(DataPoint):
    """A pitch angle (shared node)."""

    id: object = None
    name: str
    metadata: dict = {"index_fields": ["name"]}

    def __init__(self, name: str, **kwargs):
        super().__init__(id=_deterministic_id("PitchAngle", name), name=name, **kwargs)


class OutcomeNode(DataPoint):
    """CLOSED_WON or CLOSED_LOST (shared node)."""

    id: object = None
    name: str
    metadata: dict = {"index_fields": ["name"]}

    def __init__(self, name: str, **kwargs):
        super().__init__(id=_deterministic_id("Outcome", name), name=name, **kwargs)


class CustomerProblemNode(DataPoint):
    """A customer problem category (shared node — e.g. 'scaling_ai')."""

    id: object = None
    name: str
    description: str = ""
    metadata: dict = {"index_fields": ["name", "description"]}

    def __init__(self, name: str, description: str = "", **kwargs):
        super().__init__(
            id=_deterministic_id("CustomerProblem", name),
            name=name,
            description=description,
            **kwargs,
        )


class SalesTraceNode(DataPoint):
    """A full conversation trace with edges to shared feature/angle/outcome/problem nodes.

    Graph structure created by add_data_points:
      SalesTrace --customer_problem--> CustomerProblemNode
      SalesTrace --features_pitched--> SalesFeatureNode (one per feature tried)
      SalesTrace --winning_feature--> SalesFeatureNode (only if won)
      SalesTrace --winning_angle--> PitchAngleNode (only if won)
      SalesTrace --outcome--> OutcomeNode
    """

    text: str  # summary for vector search
    customer_problem: CustomerProblemNode
    features_pitched: list[SalesFeatureNode]
    winning_feature: list[SalesFeatureNode] = []  # empty list if lost, 1 item if won
    winning_angle: list[PitchAngleNode] = []  # empty list if lost, 1 item if won
    outcome: OutcomeNode
    belongs_to_set: list = [
        NodeSet(
            id=uuid5(NAMESPACE_OID, "NodeSet:sales_traces"),
            name="sales_traces",
        )
    ]
    metadata: dict = {"index_fields": ["text"]}


SALES_COGNIFY_PROMPT = (
    "Extract entities and relationships from this sales conversation summary. "
    "Focus on:\n"
    "- The customer's stated problem or need (e.g., 'data scattered across systems', "
    "'agents repeat mistakes', 'need per-user isolation')\n"
    "- Cognee features that were pitched (e.g., feedback, memory, access_control)\n"
    "- Pitch angles used (e.g., developer_experience, compliance, simplicity)\n"
    "- The outcome (CLOSED_WON or CLOSED_LOST)\n"
    "- Which feature+angle combination led to the outcome\n\n"
    "Create edges like:\n"
    "- customer_problem --solved_by--> feature\n"
    "- feature --framed_as--> angle\n"
    "- customer_problem --resulted_in--> outcome\n"
    "- customer_problem --closed_with--> feature (only for CLOSED_WON)\n"
)
