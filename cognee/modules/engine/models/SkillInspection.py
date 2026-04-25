"""Inspection models for analyzing skill failures."""

from typing import List, Literal

from pydantic import BaseModel, Field

from cognee.low_level import DataPoint


class InspectionResult(BaseModel):
    """LLM response model for skill failure inspection (not persisted)."""

    failure_category: Literal[
        "instruction_gap",
        "ambiguity",
        "wrong_scope",
        "tooling",
        "context_missing",
        "other",
    ] = Field(description="Category of the root failure")
    root_cause: str = Field(description="Concise description of the root cause")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="How severe the failure is"
    )
    improvement_hypothesis: str = Field(description="Actionable hypothesis for improving the skill")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SkillInspection(DataPoint):
    """Persisted inspection node in graph."""

    inspection_id: str
    skill_id: str
    skill_name: str
    failure_category: str
    root_cause: str
    severity: str
    improvement_hypothesis: str
    analyzed_run_ids: List[str] = Field(default_factory=list)
    analyzed_run_count: int = 0
    avg_success_score: float = 0.0
    inspection_model: str = ""
    inspection_confidence: float = 0.0

    metadata: dict = Field(
        default_factory=lambda: {"index_fields": ["root_cause", "improvement_hypothesis"]}
    )
