from uuid import UUID
from pydantic import BaseModel
from typing import Any, Optional


class SearchResultDataset(BaseModel):
    id: UUID
    name: str


class SearchResult(BaseModel):
    search_result: Any
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
