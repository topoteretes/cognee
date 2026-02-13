from typing import Optional

from pydantic import BaseModel


class FeedbackDetectionResult(BaseModel):
    """
    Result of analyzing a user message for feedback about a previous response.

    - feedback_detected: True if the message is (wholly or partly) feedback about the last answer.
    - feedback_text: Optional extracted or normalized feedback text.
    - feedback_score: Optional numeric score (e.g. 1-5); normalized to int 1-5 when persisting.
    """

    feedback_detected: bool = False
    feedback_text: Optional[str] = None
    feedback_score: Optional[float] = None
