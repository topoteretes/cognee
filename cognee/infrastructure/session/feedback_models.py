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
        description="When feedback_detected is True: required, must never be empty. Short description that includes or summarizes the user's message (e.g. 'User gave a positive rating' for '5/5').",
    )
    feedback_score: Optional[float] = Field(
        default=None,
        description="When feedback_detected is True: required. Numeric score 1-5 (1=negative, 5=positive); normalized to int when persisting.",
    )
    response_to_user: Optional[str] = Field(
        default=None,
        description="When feedback_detected is True: required. Brief, friendly message to show the user (e.g. thanking them for feedback). One sentence; can adapt tone or language to the user's message.",
    )
