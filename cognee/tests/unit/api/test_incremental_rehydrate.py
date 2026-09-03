"""Unit tests for chunk rehydration during incremental-update renumbering.

Graph adapters replace a node's whole property set on MERGE, so a rebuilt
chunk must carry EVERY model field from the stored node — anything missing is
erased, not merely reset (the bug: renumbering reset importance_weight and
truth fields to model defaults).
"""

from uuid import uuid4

from cognee.api.v1.update.incremental import _misindexed_chunks, _rehydrate_chunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument


def _document() -> TextDocument:
    return TextDocument(
        id=uuid4(),
        title="doc.txt",
        raw_data_location="/tmp/doc.txt",
        name="doc",
        mime_type="text/plain",
        external_metadata="{}",
        importance_weight=0.5,
    )


def _stored_node(text: str, chunk_index: int, **extra) -> dict:
    node = {
        "id": str(uuid4()),
        "text": text,
        "chunk_size": 42,
        "chunk_index": chunk_index,
        "cut_type": "paragraph_end",
        "content_hash": None,
        "max_chunk_tokens": 60,
        "importance_weight": 0.91,
        "truth_alignment": [0.1, 0.2],
        "truth_epoch": 7,
        "ontology_valid": True,
        "ontology_uri": "http://onto/x",
        "version": 3,
        "topological_rank": 5,
    }
    node.update(extra)
    return node


def test_rehydrate_preserves_every_model_field():
    document = _document()
    node = _stored_node("chunk text ", 4)

    chunk = _rehydrate_chunk(document, node, chunk_index=9)

    assert str(chunk.id) == node["id"]
    assert chunk.chunk_index == 9  # the ONLY field meant to change
    assert chunk.text == "chunk text "
    assert chunk.chunk_size == 42
    assert chunk.importance_weight == 0.91
    assert chunk.truth_alignment == [0.1, 0.2]
    assert chunk.truth_epoch == 7
    assert chunk.ontology_valid is True
    assert chunk.ontology_uri == "http://onto/x"
    assert chunk.version == 3
    assert chunk.topological_rank == 5
    assert chunk.cut_type == "paragraph_end"
    assert chunk.content_hash  # backfilled from text when the node lacks it
    assert chunk.max_chunk_tokens == 60


def test_rehydrate_tolerates_missing_and_malformed_fields():
    document = _document()
    node = {
        "id": str(uuid4()),
        "text": "bare chunk",
        "truth_alignment": "not-a-list",  # some adapters serialize oddly
    }
    chunk = _rehydrate_chunk(document, node, chunk_index=0)
    assert chunk.truth_alignment is None
    assert chunk.importance_weight == document.importance_weight
    assert chunk.version == 1
    assert chunk.max_chunk_tokens is None  # legacy nodes have no recorded budget


def test_misindexed_chunks_repairs_only_the_drifted_ones():
    """Self-heal for stored order left inconsistent by an interrupted run.

    A crash between a delete and its renumbering leaves survivors carrying
    stale indexes. Only those are rebuilt — a chunk already sitting at its
    actual position must not be rewritten for nothing.
    """
    document = _document()
    stored = [
        _stored_node("a ", 0),  # correct
        _stored_node("b ", 5),  # drifted: actually at position 1
        _stored_node("c ", 2),  # correct
        _stored_node("d ", 9),  # drifted: actually at position 3
    ]

    repaired = _misindexed_chunks(document, stored)

    assert [str(chunk.id) for chunk in repaired] == [stored[1]["id"], stored[3]["id"]]
    assert [chunk.chunk_index for chunk in repaired] == [1, 3]
    assert all(chunk.importance_weight == 0.91 for chunk in repaired)  # preserved


def test_a_consistent_document_needs_no_repair():
    document = _document()
    stored = [_stored_node("a ", 0), _stored_node("b ", 1), _stored_node("c ", 2)]

    assert _misindexed_chunks(document, stored) == []


def test_planned_moves_rehydrate_at_their_target_position():
    """The writer rebuilds a survivor at the position the plan assigned it.

    Rehydration is the writer's job, not the policy's: it carries every stored
    field across, and adapters replace a node's whole property set on MERGE,
    so a field left out would be erased rather than reset.
    """
    document = _document()
    node = _stored_node("c ", 2)

    moved = _rehydrate_chunk(document, node, 4)

    assert str(moved.id) == node["id"]  # identity survives the move
    assert moved.chunk_index == 4
    assert moved.text == node["text"]
    assert moved.importance_weight == 0.91
