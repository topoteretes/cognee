from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FeedbackDetectionResult(BaseModel):
    """
    Result of analyzing a user message for feedback about a previous response.
    """

    feedback_detected: bool = Field(
        default=False,
        description="True if the message is (wholly or partly) feedback about the last answer.",
    )
    feedback_text: str | None = Field(
        default=None,
        description="When feedback_detected is True: required, must never be empty. Short description that includes or summarizes the user's message (e.g. 'User gave a positive rating' for '5/5').",
    )
    feedback_score: float | None = Field(
        default=None,
        description="When feedback_detected is True: required. Numeric score 1-5 (1=negative, 5=positive); normalized to int when persisting.",
    )
    response_to_user: str | None = Field(
        default=None,
        description="When feedback_detected is True: required. Brief, friendly message to show the user (e.g. thanking them for feedback). One sentence; can adapt tone or language to the user's message.",
    )
    contains_followup_question: bool = Field(
        default=False,
        description="True if the message contains both feedback and a new or follow-up question that should be answered (e.g. 'that was wrong, but what about X?'). Set to false when the message is only feedback with no question.",
    )


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
