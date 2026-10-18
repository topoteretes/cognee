from __future__ import annotations

from pydantic import Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Skill import Skill
from cognee.modules.engine.models.SkillRun import SkillRun


class SkillImprovementProposal(DataPoint):
    """Reviewable graph-only proposal for improving a stored Skill."""

    proposal_id: str
    skill_id: str
    skill_name: str
    skill: Skill | None = None
    dataset_scope: list[str] = Field(default_factory=list)
    old_procedure: str = ""
    proposed_procedure: str = ""
    runs_used: list[str] = Field(default_factory=list)
    runs: list[SkillRun] = Field(default_factory=list)
    model_name: str = ""
    confidence: float = 0.0
    rationale: str = ""
    status: str = "proposed"

    metadata: dict = Field(
        default={
            "index_fields": ["skill_name", "rationale"],
            "identity_fields": ["proposal_id"],
        }
    )
