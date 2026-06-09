from typing import Any, List

from pydantic import BaseModel, Field, field_validator, model_validator

from cognee.infrastructure.session.session_context_models import (
    CandidateContextUpdate,
    ServedContextRating,
)


class PreviousAnswerFeedback(BaseModel):
    """Compatibility feedback to persist on the previously stored QA entry."""

    feedback_text: str = Field(
        description="Short description that includes or summarizes the user's feedback.",
    )
    feedback_score: float | None = Field(
        default=None,
        description="Numeric score 1-5 (1=negative, 5=positive); normalized when persisting.",
    )

    @field_validator("feedback_text")
    @classmethod
    def feedback_text_non_empty(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("feedback_text must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("feedback_text must be a non-empty string")
        return stripped

    @field_validator("feedback_score")
    @classmethod
    def feedback_score_clamp(cls, value: float | None) -> float | None:
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            raise ValueError("feedback_score must be a number")
        return max(1.0, min(5.0, score))


class SessionTurnAnalysis(BaseModel):
    """
    Result of analyzing a user message for session feedback and durable guidance.
    """

    previous_answer_feedback: PreviousAnswerFeedback | None = Field(
        default=None,
        description="Feedback about the previous answer, if the user gave any.",
    )
    response_to_user: str | None = Field(
        default=None,
        description="Brief acknowledgement to show the user when feedback is detected.",
    )
    contains_followup_question: bool = Field(
        default=False,
        description="True when the message contains both feedback and a new/follow-up question.",
    )
    followup_question: str | None = Field(
        default=None,
        description="The new/follow-up question text when contains_followup_question is true.",
    )
    served_context_ratings: List[ServedContextRating] = Field(
        default_factory=list,
        description="Up to 3 ratings of session-context entries served to the previous answer.",
    )
    candidate_context_updates: List[CandidateContextUpdate] = Field(
        default_factory=list,
        description="Up to 3 proposed new guidance entries.",
    )

    @field_validator("served_context_ratings", "candidate_context_updates", mode="after")
    @classmethod
    def cap_three(cls, v: list) -> list:
        """Truncate the list to the first 3 items; never raises (non-blocking contract)."""
        if not isinstance(v, list):
            return []
        return v[:3]

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_feedback_fields(cls, data: Any) -> Any:
        """Accept the old FeedbackDetectionResult shape while call sites are being rewired."""
        if not isinstance(data, dict):
            return data
        if "previous_answer_feedback" in data:
            return data
        if not data.get("feedback_detected"):
            return data
        feedback_text = data.get("feedback_text")
        if not isinstance(feedback_text, str) or not feedback_text.strip():
            return data
        data = dict(data)
        data["previous_answer_feedback"] = {
            "feedback_text": feedback_text,
            "feedback_score": data.get("feedback_score"),
        }
        return data

    @property
    def feedback_detected(self) -> bool:
        return self.previous_answer_feedback is not None

    @property
    def feedback_text(self) -> str | None:
        if self.previous_answer_feedback is None:
            return None
        return self.previous_answer_feedback.feedback_text

    @property
    def feedback_score(self) -> float | None:
        if self.previous_answer_feedback is None:
            return None
        return self.previous_answer_feedback.feedback_score


# Backwards-compatible import name while external/internal callers migrate.
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
