from typing import Optional

from pydantic import BaseModel, Field


class FeedbackDetectionResult(BaseModel):
    """
    Result of analyzing a user message for feedback about a previous response.
    """

    feedback_detected: bool = Field(
        default=False,
        description="True if the message is (wholly or partly) feedback about the last answer.",
    )
    feedback_text: Optional[str] = Field(
        default=None,
        description="When feedback_detected is True: required. Extracted or normalized feedback text (e.g. summary of what the user said). Use empty string if the message is only a score or very short.",
    )
    feedback_score: Optional[float] = Field(
        default=None,
        description="When feedback_detected is True: required. Numeric score 1-5 (1=negative, 5=positive); normalized to int when persisting.",
    )
