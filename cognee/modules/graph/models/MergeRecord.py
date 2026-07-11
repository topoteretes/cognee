from typing import List, Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from .FieldResolution import FieldResolution

class MergeRecord(DataPoint):
    """Audit trail: which nodes merged, why, and what changed."""
    survivor_id: UUID
    absorbed_id: UUID
    merge_reason: str
    field_resolutions: List[FieldResolution] = []
    reversible: bool = True
    pipeline_run_id: Optional[UUID] = None
    # NO identity_fields -> UUID4 -> re-cognify creates a new record, 
    # not overwriting the old one.
    metadata: dict = {"index_fields": []}
