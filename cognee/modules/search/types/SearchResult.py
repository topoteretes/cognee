from uuid import UUID
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class SearchResultDataset(BaseModel):
    id: UUID
    name: str


class CombinedSearchResult(BaseModel):
    result: Optional[Any]
    context: Dict[str, Any]
    graphs: Optional[Dict[str, Any]] = {}
    datasets: Optional[List[SearchResultDataset]] = None


class SearchResult(BaseModel):
    search_result: Any
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
