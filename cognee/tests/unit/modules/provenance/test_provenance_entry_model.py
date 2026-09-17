"""Pydantic <-> ORM roundtrip and defaults for ProvenanceEntry."""

from datetime import datetime

from cognee.modules.provenance.models import ProvenanceEntry, ProvenanceEntryRow


def _full_entry() -> ProvenanceEntry:
    return ProvenanceEntry(
        entity_id="entity-1",
        entity_type="entity",
        activity_id="cognify:run-1",
        agent_id="user@example.com",
        role="generator",
        source_document="/tmp/doc.txt",
        source_location="page 2",
        source_quote="quoted text",
        source_ref_key="source_ref:v1:ds:data",
        first_seen="2026-08-14T00:00:00+00:00",
        last_updated="2026-08-14T00:00:01+00:00",
        confidence=0.9,
        credibility=0.8,
        checksum="ab" * 32,
        sequence_id=7,
        previous_checksum="cd" * 32,
        parent_entity_id="parent-1",
        used_entities=["parent-1", "other-1"],
        previous_version_id="entity-1:v:x",
        derived_from_id="parent-1",
        acted_on_behalf_of="org-1",
        informed_by_activities=["cognify:run-0"],
        revision_type="correction",
        supersedes="entity-0",
        bundle_id="run-1",
        invalidated=True,
        invalidated_at_time="2026-08-14T00:00:02+00:00",
        invalidated_by="auditor",
        invalidation_reason="retracted",
        start_index=3,
        end_index=4,
        metadata={"name": "Alice", "nested": {"k": [1, 2]}},
    )


class TestRoundtrip:
    def test_row_roundtrip_preserves_every_field(self):
        entry = _full_entry()
        row = entry.to_row()
        restored = ProvenanceEntry.from_row(row)
        assert restored.model_dump() == entry.model_dump()

    def test_metadata_maps_to_entry_metadata_column(self):
        entry = _full_entry()
        row = entry.to_row()
        # Attribute is entry_metadata; the actual column name stays "metadata".
        assert row.entry_metadata == entry.metadata
        assert "metadata" in ProvenanceEntryRow.__table__.c
        assert ProvenanceEntryRow.entry_metadata.expression.name == "metadata"

    def test_json_list_and_dict_fields_survive(self):
        entry = _full_entry()
        restored = ProvenanceEntry.from_row(entry.to_row())
        assert restored.used_entities == ["parent-1", "other-1"]
        assert restored.informed_by_activities == ["cognify:run-0"]
        assert restored.metadata["nested"] == {"k": [1, 2]}

    def test_apply_to_row_overwrites_in_place(self):
        row = _full_entry().to_row()
        updated = _full_entry().model_copy(update={"agent_id": "other", "sequence_id": 9})
        updated.apply_to_row(row)
        assert row.agent_id == "other"
        assert row.sequence_id == 9


class TestDefaults:
    def test_defaults(self):
        entry = ProvenanceEntry(entity_id="x")
        assert entry.entity_type == "entity"
        assert entry.agent_id == "cognee"
        assert entry.agent_type == "software_agent"
        assert entry.is_automated is True
        assert entry.confidence == 1.0
        assert entry.used_entities == []
        assert entry.informed_by_activities == []
        assert entry.metadata == {}
        assert entry.invalidated is False
        assert entry.version == "1.0"
        assert entry.sequence_id is None
        assert entry.checksum is None

    def test_default_timestamp_is_timezone_aware_iso(self):
        entry = ProvenanceEntry(entity_id="x")
        parsed = datetime.fromisoformat(entry.timestamp)
        assert parsed.tzinfo is not None
