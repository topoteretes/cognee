from uuid import UUID
from typing import Any, Dict
from pydantic import BaseModel

class ScoredResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]
