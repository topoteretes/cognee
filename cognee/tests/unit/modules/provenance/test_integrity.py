"""Checksum canonicalization and tamper detection (pure, no DB)."""

import pytest

from cognee.modules.provenance.integrity import (
    _HASHED_FIELDS,
    canonical_entity_id,
    compute_checksum,
    verify_checksum,
)
from cognee.modules.provenance.models import ProvenanceEntry

_LAST_UPDATED = "2026-08-14T00:00:00+00:00"


def _entry(**overrides) -> ProvenanceEntry:
    values = {
        "entity_id": "entity-1",
        "entity_type": "entity",
        "activity_id": "cognify:run-1",
        "source_document": "/tmp/doc.txt",
        "timestamp": "2026-08-14T00:00:00+00:00",
        "last_updated": _LAST_UPDATED,
        "used_entities": ["a", "b"],
    }
    values.update(overrides)
    return ProvenanceEntry(**values)


class TestCanonicalization:
    def test_deterministic(self):
        assert compute_checksum(_entry()) == compute_checksum(_entry())

    def test_separator_prevents_field_bleed(self):
        # Without a separator "ab"+"c" and "a"+"bc" would concatenate equal.
        first = _entry(entity_type="ab", activity_id="c")
        second = _entry(entity_type="a", activity_id="bc")
        assert compute_checksum(first) != compute_checksum(second)

    def test_separator_inside_a_value_cannot_bleed(self):
        # JSON-encoding each part escapes \x1f, so content cannot move across
        # a field boundary and hash the same.
        first = _entry(invalidated_by="alice", invalidation_reason="reason")
        second = _entry(invalidated_by="alice\x1freason", invalidation_reason="")
        assert compute_checksum(first) != compute_checksum(second)

    def test_list_element_boundaries_are_unambiguous(self):
        assert compute_checksum(_entry(used_entities=["a,b"])) != compute_checksum(
            _entry(used_entities=["a", "b"])
        )

    def test_none_and_empty_string_are_distinct(self):
        # JSON canonicalization: null vs "" are different audit states.
        assert compute_checksum(_entry(parent_entity_id=None)) != compute_checksum(
            _entry(parent_entity_id="")
        )

    def test_entity_id_relabel_does_not_change_checksum(self):
        # The archival-relabel invariant: the archive suffix (which embeds the
        # row's own last_updated) canonicalizes away.
        entry = _entry()
        for suffix in ("", ":3"):
            relabeled = entry.model_copy(
                update={"entity_id": f"entity-1:v:{_LAST_UPDATED}{suffix}"}
            )
            assert canonical_entity_id(relabeled) == "entity-1"
            assert compute_checksum(entry) == compute_checksum(relabeled)

    def test_entity_id_swap_changes_checksum(self):
        # Reattributing a row to another entity must break verification.
        entry = _entry()
        swapped = entry.model_copy(update={"entity_id": "entity-2"})
        assert compute_checksum(entry) != compute_checksum(swapped)

    def test_hashed_fields_cover_the_whole_entry(self):
        # Every column except the hash output participates (entity_id joins in
        # canonical form). A field added to the model must be added to the
        # frozen contract deliberately.
        expected = set(ProvenanceEntry.model_fields) - {"entity_id", "checksum"}
        assert set(_HASHED_FIELDS) == expected


_TAMPER_VALUES = {
    "used_entities": ["a", "b", "tampered"],
    "informed_by_activities": ["tampered"],
    "invalidated": True,
    "is_automated": False,
    "confidence": 0.123,
    "credibility": 0.5,
    "sequence_id": 999,
    "start_index": 41,
    "end_index": 43,
    "metadata": {"tampered": True},
}


class TestVerifyChecksum:
    def test_verify_roundtrip(self):
        entry = _entry()
        entry.checksum = compute_checksum(entry)
        assert verify_checksum(entry) is True

    def test_missing_checksum_is_invalid(self):
        assert verify_checksum(_entry()) is False

    @pytest.mark.parametrize("field", _HASHED_FIELDS)
    def test_editing_any_hashed_field_fails_verification(self, field):
        entry = _entry(
            parent_entity_id="p",
            previous_version_id="v",
            derived_from_id="d",
            previous_checksum="prev",
            sequence_id=7,
            start_index=1,
            end_index=2,
            metadata={"kind": "test"},
            invalidated_at_time="2026-08-14T00:00:01+00:00",
            invalidated_by="auditor",
            invalidation_reason="reason",
        )
        entry.checksum = compute_checksum(entry)
        tampered = entry.model_copy(update={field: _TAMPER_VALUES.get(field, f"tampered-{field}")})
        tampered.checksum = entry.checksum
        assert verify_checksum(tampered) is False
