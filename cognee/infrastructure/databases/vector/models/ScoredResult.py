from uuid import UUID
from typing import Any, Dict
from pydantic import BaseModel

class ScoredResult(BaseModel):
    id: UUID
    score: int
    payload: Dict[str, Any]
