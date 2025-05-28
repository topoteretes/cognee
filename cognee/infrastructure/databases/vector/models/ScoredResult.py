from typing import Any, Dict
from uuid import UUID
from pydantic import BaseModel


class ScoredResult(BaseModel):
    """
    Represents the result of a scoring operation with an identification and associated data.

    Attributes:

    - id (UUID): Unique identifier for the scored result.
    - score (float): The score associated with the result, where a lower score indicates a
    better outcome.
    - payload (Dict[str, Any]): Additional information related to the score, stored as
    key-value pairs in a dictionary.
    """

    id: UUID
    score: float  # Lower score is better
    payload: Dict[str, Any]
