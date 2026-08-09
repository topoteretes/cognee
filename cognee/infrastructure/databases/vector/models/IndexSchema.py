"""Shared vector-index payload model and conversion helper."""

from typing import List, Optional

from cognee.infrastructure.engine import DataPoint


class IndexSchema(DataPoint):
    """The backend-neutral payload stored for a vector index row."""

    text: str

    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    source_chunk_id: Optional[str] = None
    importance_weight: Optional[float] = 0.5
    number_of_edges: Optional[int] = None
    relationship_name: Optional[str] = None
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: List[str] = []


def index_schema_from_data_point(data_point: DataPoint) -> IndexSchema:
    """Convert any indexable point without discarding optional vector payload."""
    fields = (
        "document_id",
        "document_name",
        "chunk_index",
        "source_chunk_id",
        "importance_weight",
        "number_of_edges",
        "relationship_name",
        "source_node_id",
        "target_node_id",
    )
    return IndexSchema(
        id=data_point.id,
        text=DataPoint.get_embeddable_data(data_point),
        belongs_to_set=data_point.belongs_to_set or [],
        **{field: getattr(data_point, field, None) for field in fields},
    )
