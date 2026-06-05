from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

VALID_SECTIONS = {"goals", "rules", "preferences", "lessons_learned"}
VALID_RATINGS = {"helpful", "harmful"}
MAX_CONTEXT_CONTENT_CHARS = 280
MIN_CANDIDATE_CONFIDENCE = 0.75


class ContextSection(str, Enum):
    """Sections under which active session-context guidance is grouped."""

    GOALS = "goals"
    RULES = "rules"
    PREFERENCES = "preferences"
    LESSONS_LEARNED = "lessons_learned"


def normalize_content(text: str) -> str:
    """Deterministic exact-match key: lowercased, whitespace-collapsed, stripped."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return " ".join(text.strip().lower().split())


class ServedContextRating(BaseModel):
    """A single rating of a session-context entry that was served to the previous answer."""

    entry_id: str
    rating: str

    @field_validator("entry_id")
    @classmethod
    def entry_id_non_empty(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("entry_id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("entry_id must be a non-empty string")
        return stripped

    @field_validator("rating")
    @classmethod
    def rating_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("rating must be a string")
        normalized = v.strip().lower()
        if normalized not in VALID_RATINGS:
            raise ValueError(f"rating must be one of {sorted(VALID_RATINGS)}")
        return normalized


class CandidateContextUpdate(BaseModel):
    """A proposed new active session-context guidance entry emitted by feedback detection."""

    section: str
    content: str
    confidence: float = 0.0

    @field_validator("section")
    @classmethod
    def section_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("section must be a string")
        normalized = v.strip().lower()
        if normalized not in VALID_SECTIONS:
            raise ValueError(f"section must be one of {sorted(VALID_SECTIONS)}")
        return normalized

    @field_validator("content")
    @classmethod
    def content_truncate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        stripped = v.strip()
        return stripped[:MAX_CONTEXT_CONTENT_CHARS]

    @field_validator("confidence")
    @classmethod
    def confidence_clamp(cls, v: float) -> float:
        if v is None:
            return 0.0
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number")
        return max(0.0, min(1.0, value))


class SessionContextEntry(BaseModel):
    """A stored active session-context guidance entry (goal/rule/preference/lesson)."""

    id: str
    section: str
    content: str
    normalized_content: str
    confidence: float = 0.0
    created_at: str
    source_feedback_ids: List[str] = Field(default_factory=list)
    helpful_count: int = 0
    harmful_count: int = 0
    priority: int = 0
    last_served_at: Optional[str] = None
    kind: Literal["context", "feedback"] = "context"

    @field_validator("section")
    @classmethod
    def section_valid(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("section must be a string")
        normalized = v.strip().lower()
        if normalized not in VALID_SECTIONS:
            raise ValueError(f"section must be one of {sorted(VALID_SECTIONS)}")
        return normalized

    @field_validator("content")
    @classmethod
    def content_non_empty_truncate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must be a non-empty string")
        return stripped[:MAX_CONTEXT_CONTENT_CHARS]

    @field_validator("confidence")
    @classmethod
    def confidence_clamp(cls, v: float) -> float:
        if v is None:
            return 0.0
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number")
        return max(0.0, min(1.0, value))

    @field_validator("helpful_count", "harmful_count")
    @classmethod
    def counts_non_negative(cls, v: int) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("count must be an integer")
        if v < 0:
            raise ValueError("count must be >= 0")
        return v


class SessionFeedbackEntry(BaseModel):
    """A stored feedback record describing how a turn rated/extended session context."""

    id: str
    created_at: str
    raw_text: str
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None
    referenced_qa_ids: List[str] = Field(default_factory=list)
    influencing_context_ids: List[str] = Field(default_factory=list)
    candidate_context_entries: List[dict] = Field(default_factory=list)
    kind: Literal["context", "feedback"] = "feedback"

    @field_validator("feedback_score")
    @classmethod
    def feedback_score_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 5):
            raise ValueError("feedback_score must be between 1 and 5")
        return v
