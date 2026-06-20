import pytest

from cognee.infrastructure.databases.vector.adapters.chromadb.ChromaDBAdapter import (
    BELONGS_TO_SET_KEY,
    LEGACY_BELONGS_TO_SET_LIST_KEY,
    LEGACY_BELONGS_TO_SET_MEMBER_PREFIX,
    ChromaDBAdapter,
    metadata_needs_belongs_to_set_migration,
    migrate_belongs_to_set_metadata,
    process_data_for_chroma,
    restore_data_from_chroma,
    sanitize_node_names,
)


def test_process_data_for_chroma_stores_native_belongs_to_set_array():
    processed = process_data_for_chroma({"belongs_to_set": ["alpha", "beta"]})
    assert processed["belongs_to_set"] == ["alpha", "beta"]
    assert LEGACY_BELONGS_TO_SET_LIST_KEY not in processed


def test_process_data_for_chroma_still_json_encodes_other_lists():
    processed = process_data_for_chroma({"tags": ["x"]})
    assert "tags__list" in processed


def test_restore_data_from_chroma_reads_native_array():
    restored = restore_data_from_chroma({"belongs_to_set": ["alpha"]})
    assert restored["belongs_to_set"] == ["alpha"]


def test_restore_data_from_chroma_reads_legacy_json_list():
    restored = restore_data_from_chroma(
        {LEGACY_BELONGS_TO_SET_LIST_KEY: '["legacy-tag"]'}
    )
    assert restored["belongs_to_set"] == ["legacy-tag"]


def test_migrate_belongs_to_set_metadata_converts_legacy_keys():
    metadata = {
        LEGACY_BELONGS_TO_SET_LIST_KEY: '["a"]',
        f"{LEGACY_BELONGS_TO_SET_MEMBER_PREFIX}b": True,
    }
    migrated = migrate_belongs_to_set_metadata(metadata)
    assert migrated[BELONGS_TO_SET_KEY] == ["a", "b"]
    assert LEGACY_BELONGS_TO_SET_LIST_KEY not in migrated
    assert not any(key.startswith(LEGACY_BELONGS_TO_SET_MEMBER_PREFIX) for key in migrated)


def test_metadata_needs_belongs_to_set_migration_detects_legacy_formats():
    assert metadata_needs_belongs_to_set_migration({LEGACY_BELONGS_TO_SET_LIST_KEY: "[]"})
    assert metadata_needs_belongs_to_set_migration(
        {f"{LEGACY_BELONGS_TO_SET_MEMBER_PREFIX}tag": True}
    )
    assert not metadata_needs_belongs_to_set_migration({"belongs_to_set": ["native"]})


def test_build_where_filter_uses_contains_operator():
    where = ChromaDBAdapter._build_where_filter(["alpha"], "OR")
    assert where == {"belongs_to_set": {"$contains": "alpha"}}


def test_build_where_filter_and_semantics():
    where = ChromaDBAdapter._build_where_filter(["alpha", "beta"], "AND")
    assert where == {
        "$and": [
            {"belongs_to_set": {"$contains": "alpha"}},
            {"belongs_to_set": {"$contains": "beta"}},
        ]
    }


def test_sanitize_node_names_rejects_blank_and_overlong_values():
    assert sanitize_node_names([" valid ", "", "x" * 300]) == ["valid"]
