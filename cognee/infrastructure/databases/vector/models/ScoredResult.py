from typing import Any, Dict
from uuid import UUID
from pydantic import BaseModel


class ScoredResult(BaseModel):
    id: UUID
    score: float  # Lower score is better
    payload: Dict[str, Any]
