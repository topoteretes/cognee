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


def index_schema_from_data_point(
    data_point: DataPoint, index_property_name: Optional[str] = None
) -> IndexSchema:
    """Convert an indexable point while retaining the collection's source text."""
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
    text = (
        getattr(data_point, index_property_name, None)
        if index_property_name is not None
        else DataPoint.get_embeddable_data(data_point)
    )
    if isinstance(text, str):
        text = text.strip()

    return IndexSchema(
        id=data_point.id,
        text=text,
        belongs_to_set=data_point.belongs_to_set or [],
        **{field: getattr(data_point, field, None) for field in fields},
    )
