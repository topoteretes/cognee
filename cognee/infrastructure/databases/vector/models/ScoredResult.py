from typing import Any, Dict
from pydantic import BaseModel

class ScoredResult(BaseModel):
    id: str
    score: float # Lower score is better
    payload: Dict[str, Any]
