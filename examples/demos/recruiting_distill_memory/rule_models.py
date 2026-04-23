"""Pydantic models for the Ledgerline recruiting demo.

Rule is the graph-node schema for the rulebook: seed rules ingested from
Alex's playbook and, later, any agent-proposed entries a human approves.

Tool outputs are plain domain objects. They carry no citation fields
(applied_rule_ids, compliance_notes, proposed_new_rules) — the tools are
not aware they're being observed. The decorator captures traces and the
memify pipeline extracts learnings from those traces post-hoc.
"""

from typing import Annotated, Literal

from pydantic import BaseModel

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


class InterviewFormatProposal(BaseModel):
    format: Literal["live_coding", "take_home", "system_design", "behavioral_only"]
    duration_minutes: int
    medium: Literal["video", "onsite", "phone"]


class PanelSchedule(BaseModel):
    panelists: list[str]
    total_hours: float
    cto_included: bool


class ScreenInvite(BaseModel):
    subject: str
    body: str
    disclosure_questions: list[str]
