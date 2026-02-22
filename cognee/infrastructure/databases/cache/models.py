from typing import Optional
from pydantic import BaseModel, field_validator


class SessionQAEntry(BaseModel):
    """
    Canonical format for a QA entry stored in session cache.

    Fields:
        time: ISO format timestamp when the QA was created.
        qa_id: Unique identifier for the entry (required for update/delete).
        question: User's question.
        context: Context used to generate the answer.
        answer: Generated answer.
        feedback_text: Optional user feedback text.
        feedback_score: Optional feedback score 1-5.
    """

    time: str
    question: str
    context: str
    answer: str
    qa_id: Optional[str] = None
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None

    @field_validator("feedback_score")
    @classmethod
    def feedback_score_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 5):
            raise ValueError("feedback_score must be between 1 and 5")
        return v
