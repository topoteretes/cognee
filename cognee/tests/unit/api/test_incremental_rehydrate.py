"""Unit tests for chunk rehydration during incremental-update renumbering.

Graph adapters replace a node's whole property set on MERGE, so a rebuilt
chunk must carry EVERY model field from the stored node — anything missing is
erased, not merely reset (the bug: renumbering reset importance_weight and
truth fields to model defaults).
"""

from uuid import uuid4

from cognee.api.v1.update.incremental import _build_shifted_chunks, _rehydrate_chunk
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


def test_shifted_chunks_rehydrate_and_renumber_only_moved_survivors():
    document = _document()
    stored = [
        _stored_node("a ", 0),
        _stored_node("b ", 1),  # affected — replaced by 3 new chunks
        _stored_node("c ", 2),  # suffix: index 2 -> 4
        _stored_node("d ", 3),  # suffix: index 3 -> 5
    ]
    # Chunk 1 was replaced by 3 new chunks: kept chunk 0 stays at 0, kept
    # chunks 2 and 3 land at final positions 4 and 5.
    shifted = _build_shifted_chunks(
        document, stored, affected={1}, kept_final_index={0: 0, 2: 4, 3: 5}
    )

    assert [str(chunk.id) for chunk in shifted] == [stored[2]["id"], stored[3]["id"]]
    assert [chunk.chunk_index for chunk in shifted] == [4, 5]
    assert all(chunk.importance_weight == 0.91 for chunk in shifted)  # preserved
    # The prefix chunk already sits at its expected index — untouched.
    assert stored[0]["id"] not in {str(chunk.id) for chunk in shifted}
