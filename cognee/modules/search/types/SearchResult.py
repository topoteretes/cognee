from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class SearchResultDataset(BaseModel):
    id: UUID
    name: str


class SearchResult(BaseModel):
    search_result: Any
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
