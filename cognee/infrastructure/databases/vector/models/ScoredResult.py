from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel


class ScoredResult(BaseModel):
    """
    Represents a vector retrieval result with an identification and associated data.

    Attributes:

    - id (UUID): Unique identifier for the scored result.
    - score (float): Raw backend distance score (cosine distance for built-in adapters), where a
    lower score indicates a better match.
    - payload (Optional[Dict[str, Any]]): Additional information related to the score, stored as
    key-value pairs in a dictionary.
    """

    id: UUID
    score: float  # Lower score is better
    payload: Optional[Dict[str, Any]] = None
