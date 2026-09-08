from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .ProvenanceEntryRow import ProvenanceEntryRow


def utc_now_iso() -> str:
    """Timezone-aware ISO-8601 timestamp (fixes semantica's naive-utcnow quirk)."""
    return datetime.now(timezone.utc).isoformat()


class ProvenanceEntry(BaseModel):
    """W3C PROV-O flavored provenance entry (API/wire shape).

    Field-for-field mirror of ``ProvenanceEntryRow``; the only mapping is the
    free-form dict, named ``metadata`` here and ``entry_metadata`` on the row
    (``metadata`` is reserved on SQLAlchemy Declarative models).

    ``entity_type`` and ``agent_type`` stay plain ``str`` for semantica
    compatibility (no enums in PR1).
    """

    model_config = ConfigDict(from_attributes=True)

    # identity / PROV-O core
    entity_id: str
    entity_type: str = "entity"
    activity_id: str = "entity_tracking"
    agent_id: str = "cognee"
    agent_type: str = "software_agent"
    is_automated: bool = True
    role: str | None = None

    # audit-grade source
    source_document: str = ""
    source_location: str | None = None
    source_quote: str | None = None
    source_ref_key: str | None = None

    # temporal
    timestamp: str = Field(default_factory=utc_now_iso)
    first_seen: str | None = None
    last_updated: str | None = None
    activity_started_at_time: str | None = None
    activity_ended_at_time: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None

    # quality / hash chain
    confidence: float = 1.0
    credibility: float | None = None
    checksum: str | None = None
    sequence_id: int | None = None
    previous_checksum: str | None = None

    # lineage
    parent_entity_id: str | None = None
    used_entities: list[str] = Field(default_factory=list)
    previous_version_id: str | None = None
    derived_from_id: str | None = None

    # governance
    acted_on_behalf_of: str | None = None
    informed_by_activities: list[str] = Field(default_factory=list)
    revision_type: str | None = None
    supersedes: str | None = None
    bundle_id: str | None = None

    # invalidation tombstone
    invalidated: bool = False
    invalidated_at_time: str | None = None
    invalidated_by: str | None = None
    invalidation_reason: str | None = None

    # chunk extras + free-form
    start_index: int | None = None
    end_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_row(self) -> ProvenanceEntryRow:
        """Build a new ORM row from this entry (metadata -> entry_metadata)."""
        values = self.model_dump()
        values["entry_metadata"] = values.pop("metadata")
        return ProvenanceEntryRow(**values)

    def apply_to_row(self, row: ProvenanceEntryRow) -> None:
        """Copy every field onto an existing ORM row (in-place UPDATE)."""
        values = self.model_dump()
        values["entry_metadata"] = values.pop("metadata")
        for name, value in values.items():
            setattr(row, name, value)

    @classmethod
    def from_row(cls, row: ProvenanceEntryRow) -> "ProvenanceEntry":
        column_names = [column.key for column in ProvenanceEntryRow.__mapper__.column_attrs]
        values = {name: getattr(row, name) for name in column_names}
        values["metadata"] = values.pop("entry_metadata") or {}
        values["used_entities"] = list(values.get("used_entities") or [])
        values["informed_by_activities"] = list(values.get("informed_by_activities") or [])
        return cls(**values)
