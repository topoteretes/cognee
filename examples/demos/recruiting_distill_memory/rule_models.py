"""Pydantic models for the Ledgerline recruiting demo.

- Rule: typed graph node for the hand-authored rulebook (status=approved) and
  agent-proposed additions (status=pending).
- ProposedRule: the shape an agent tool returns when it wants to suggest a new
  rule; not a graph node on its own — promoted to Rule(status='pending') at
  review time.
- Tool-output models (InterviewFormatProposal, PanelSchedule, ScreenInvite):
  each decorated tool returns one of these. They all carry applied_rule_ids
  and proposed_new_rules so the agent_memory decorator's trace persistence
  links actions back to rules deterministically — no post-hoc LLM judge.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cognee.infrastructure.engine import DataPoint, Dedup, Embeddable


Domain = Literal["scheduling", "screening", "offer"]
Status = Literal["approved", "pending", "rejected"]


class Rule(DataPoint):
    """A single recruiting rule — first-class graph node."""

    rule_id: Annotated[str, Dedup("stable id like R1_live_coding")]
    domain: Domain
    status: Status
    source: str
    trigger: Annotated[str, Embeddable("condition under which the rule fires")]
    action: Annotated[str, Embeddable("what to do when the rule fires")]
    rationale: Annotated[str, Embeddable("why this rule exists")]


class ProposedRule(BaseModel):
    """Shape an agent tool returns when it wants to suggest a new rule.

    Promoted to Rule(status='pending', source='agent_proposal') by the human
    review step.
    """

    rule_id: str = Field(
        description="Short id the agent coins, e.g. 'proposed_R7_remote_timezones'",
    )
    domain: Domain
    trigger: str = Field(description="Condition under which the proposed rule would fire")
    action: str = Field(description="Action the rule would prescribe")
    rationale: str = Field(description="Why the agent thinks this rule is needed")


class _ToolOutputBase(BaseModel):
    """Shared trace-linking fields for every tool return type."""

    applied_rule_ids: list[str] = Field(
        default_factory=list,
        description="rule_ids from retrieved memory that this action follows",
    )
    proposed_new_rules: list[ProposedRule] = Field(
        default_factory=list,
        description="New rules the agent proposes; land as status=pending for human review",
    )
    compliance_notes: str = Field(
        default="",
        description="One sentence: followed-as-is / extended / overrode / novel",
    )


class InterviewFormatProposal(_ToolOutputBase):
    """Output of propose_interview_format."""

    format: Literal["live_coding", "take_home", "system_design", "behavioral_only"]
    duration_minutes: int
    medium: Literal["video", "onsite", "phone"]


class PanelSchedule(_ToolOutputBase):
    """Output of schedule_panel."""

    panelists: list[str]
    total_hours: float
    cto_included: bool


class ScreenInvite(_ToolOutputBase):
    """Output of compose_screen_invite."""

    subject: str
    body: str
    disclosure_questions: list[str]
