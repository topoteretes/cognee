from typing import Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint

class ContradictionEdge(DataPoint):
    """Property-level contradiction record."""
    entity_id: UUID
    property_name: str
    old_value: str
    new_value: str
    old_source: Optional[str] = None
    new_source: Optional[str] = None
    resolved_by: str = "last_write_wins"
    # NO identity_fields -> UUID4 -> each contradiction is a distinct record
    metadata: dict = {"index_fields": []}
