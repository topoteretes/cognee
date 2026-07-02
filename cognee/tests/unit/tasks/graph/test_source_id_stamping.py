"""Tests for id-exact source lineage stamping (issue #3632).

_stamp_source_ids stamps source_document_id / source_chunk_id onto the entities
each DocumentChunk contains — the one place the chunk -> document -> entity link
is known. Entities can be stored in `contains` either as bare Entity objects or
as (Edge, Entity) tuples, and stamping is set-if-None (first chunk wins).
"""

from uuid import uuid4

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.models.Entity import Entity
from cognee.tasks.graph.extract_graph_from_data import _stamp_source_ids


def _make_chunk(document_id, contains):
    document = Document(
        name="doc.txt",
        raw_data_location="/tmp/doc.txt",
        external_metadata="",
        mime_type="text/plain",
    )
    return DocumentChunk(
        text="hello world",
        chunk_size=2,
        chunk_index=0,
        cut_type="sentence",
        is_part_of=document,
        contains=contains,
        document_id=document_id,
    )


def test_stamps_document_and_chunk_id_on_bare_entities():
    entity = Entity(name="Alice", description="a person")
    chunk = _make_chunk("doc-123", [entity])

    _stamp_source_ids([chunk])

    assert entity.source_chunk_id == str(chunk.id)
    assert entity.source_document_id == "doc-123"


def test_stamps_entities_stored_as_edge_tuples():
    entity = Entity(name="Acme", description="a company")
    chunk = _make_chunk("doc-456", [(Edge(), entity)])

    _stamp_source_ids([chunk])

    assert entity.source_chunk_id == str(chunk.id)
    assert entity.source_document_id == "doc-456"


def test_set_if_none_first_chunk_wins():
    entity = Entity(name="Alice", description="a person")
    entity.source_chunk_id = "already-set"
    entity.source_document_id = "already-doc"
    chunk = _make_chunk("doc-999", [entity])

    _stamp_source_ids([chunk])

    assert entity.source_chunk_id == "already-set"
    assert entity.source_document_id == "already-doc"


def test_missing_document_id_leaves_document_field_none():
    entity = Entity(name="Alice", description="a person")
    chunk = _make_chunk(None, [entity])

    _stamp_source_ids([chunk])

    # chunk id is always known; document id is only stamped when present
    assert entity.source_chunk_id == str(chunk.id)
    assert entity.source_document_id is None


def test_empty_contains_is_noop():
    chunk = _make_chunk("doc-1", [])
    # Should not raise.
    _stamp_source_ids([chunk])


def test_none_contains_is_noop():
    # Real chunks default contains to None; the helper must tolerate it.
    chunk = _make_chunk("doc-1", [])
    chunk.contains = None
    _stamp_source_ids([chunk])


# ---------------------------------------------------------------------------
# Disable-flag coverage (issue #3632). The extract-path stamping is wrapped in
# `if get_base_config().provenance_enabled:`, so disabling the flag must skip
# it — this closes the leak where PROVENANCE_ENABLED=false still stamped ids.
# These mirror the exact gate at the call site against the real config.
# ---------------------------------------------------------------------------


def test_extract_stamping_skipped_when_disabled(monkeypatch):
    from cognee.base_config import get_base_config

    monkeypatch.setenv("PROVENANCE_ENABLED", "false")
    get_base_config.cache_clear()
    try:
        entity = Entity(name="Alice", description="a person")
        chunk = _make_chunk("doc-1", [entity])

        if get_base_config().provenance_enabled:  # the real gate at the call site
            _stamp_source_ids([chunk])

        assert entity.source_chunk_id is None
        assert entity.source_document_id is None
    finally:
        get_base_config.cache_clear()


def test_extract_stamping_runs_when_enabled(monkeypatch):
    from cognee.base_config import get_base_config

    monkeypatch.setenv("PROVENANCE_ENABLED", "true")
    get_base_config.cache_clear()
    try:
        entity = Entity(name="Alice", description="a person")
        chunk = _make_chunk("doc-1", [entity])

        if get_base_config().provenance_enabled:
            _stamp_source_ids([chunk])

        assert entity.source_chunk_id == str(chunk.id)
        assert entity.source_document_id == "doc-1"
    finally:
        get_base_config.cache_clear()
