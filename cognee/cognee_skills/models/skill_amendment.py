"""Amendment models for skill instruction improvements."""

from pydantic import BaseModel, Field

from cognee.low_level import DataPoint


class AmendmentProposal(BaseModel):
    """LLM response model for a proposed amendment (not persisted)."""

    amended_instructions: str = Field(description="Complete amended instructions (not a diff)")
    change_explanation: str = Field(description="What was changed and why")
    expected_improvement: str = Field(
        description="What improvement is expected from this amendment"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SkillAmendment(DataPoint):
    """Persisted amendment node in graph."""

    amendment_id: str
    skill_id: str
    skill_name: str
    inspection_id: str
    original_instructions: str
    amended_instructions: str
    change_explanation: str
    expected_improvement: str
    status: str = "proposed"  # "proposed" | "applied" | "rolled_back" | "rejected"
    amendment_model: str = ""
    amendment_confidence: float = 0.0
    pre_amendment_avg_score: float = 0.0
    applied_at_ms: int = 0  # epoch ms when amendment was applied (for evaluate filtering)
    post_amendment_avg_score: float = 0.0
    post_amendment_run_count: int = 0

    metadata: dict = {"index_fields": ["change_explanation"]}
