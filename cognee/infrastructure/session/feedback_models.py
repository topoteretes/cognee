from pydantic import BaseModel, Field, field_validator

from cognee.infrastructure.session.session_context_models import (
    CandidateContextUpdate,
    ServedContextRating,
)


class SessionTurnAnalysis(BaseModel):
    """
    Result of analyzing a user message for answer routing and durable guidance.
    """

    response_to_user: str | None = Field(
        default=None,
        description="Brief acknowledgement when no answer should be generated.",
    )
    query_to_answer: str | None = Field(
        default=None,
        description="The question/request to answer now, or null when there is none.",
    )
    served_context_ratings: list[ServedContextRating] = Field(
        default_factory=list,
        description="Up to 3 ratings of session-context entries served to the previous answer.",
    )
    candidate_context_updates: list[CandidateContextUpdate] = Field(
        default_factory=list,
        description=(
            "Up to 3 proposed session-context updates. Each item has a section "
            "(goals, rules, preferences, or lessons_learned), content, and confidence."
        ),
    )

    @field_validator("response_to_user", "query_to_answer")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        stripped = value.strip()
        return stripped or None

    @field_validator("candidate_context_updates", mode="before")
    @classmethod
    def normalize_candidate_context_updates(cls, value):
        if not isinstance(value, list):
            return []

        normalized = []
        for item in value:
            if isinstance(item, CandidateContextUpdate):
                item = item.model_dump()

            if isinstance(item, dict):
                item = dict(item)
                section = item.get("section")
                if isinstance(section, str):
                    item["section"] = section.strip().lower()

            normalized.append(item)

        return normalized

    @field_validator("served_context_ratings", "candidate_context_updates", mode="after")
    @classmethod
    def cap_three(cls, v: list) -> list:
        """Truncate the list to the first 3 items; never raises (non-blocking contract)."""
        if not isinstance(v, list):
            return []
        return v[:3]


# Backwards-compatible import name for existing internal imports.
FeedbackDetectionResult = SessionTurnAnalysis


class AgentTraceFeedbackSummary(BaseModel):
    """One-sentence summary generated from an agent trace step return value."""

    session_feedback: str = Field(
        default="",
        description="One short human-readable sentence summarizing the method return value.",
    )

    @field_validator("session_feedback")
    @classmethod
    def normalize_feedback(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("session_feedback must be a string")
        return value.strip()
