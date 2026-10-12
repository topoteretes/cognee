"""Unit tests for ``remove_disconnected_chunks`` (issue #5686).

The task must resolve a document's chunks through the edges and ids the
current graph actually stores — ``is_part_of`` from the chunk to the document,
node identity ``id`` — and delete only the stored chunks of the given
documents that the given batch no longer contains. It must never sweep
unrelated isolated nodes out of the graph, and adapters that return bare id
strings instead of ``{"id": ...}`` dicts (Neo4j, Neptune) must work too.

A fake graph engine stands in for the real backend: no database is started.
"""

from importlib import import_module
from unittest.mock import AsyncMock

import pytest

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.tasks.chunks.remove_disconnected_chunks import remove_disconnected_chunks

task_module = import_module("cognee.tasks.chunks.remove_disconnected_chunks")


class FakeGraphEngine:
    """Records traversal and deletion calls; answers from a fixed map."""

    def __init__(self, predecessors_by_node=None):
        self.predecessors_by_node = predecessors_by_node or {}
        self.predecessor_queries = []
        self.successor_queries = []
        self.disconnected_nodes_queries = 0
        self.deleted_batches = []

    async def get_predecessors(self, node_id, edge_label=None):
        self.predecessor_queries.append((str(node_id), edge_label))
        return self.predecessors_by_node.get(str(node_id), [])

    async def get_successors(self, node_id, edge_label=None):
        self.successor_queries.append((str(node_id), edge_label))
        return []

    async def get_disconnected_nodes(self):
        self.disconnected_nodes_queries += 1
        return ["unrelated-isolated-node"]

    async def delete_nodes(self, node_ids):
        self.deleted_batches.append([str(node_id) for node_id in node_ids])


def _make_document(name="example.txt"):
    return Document(
        name=name,
        raw_data_location=f"/tmp/{name}",
        external_metadata=None,
        mime_type="text/plain",
    )


def _make_chunk(document, text):
    return DocumentChunk(
        text=text,
        chunk_size=len(text),
        chunk_index=0,
        cut_type="paragraph",
        is_part_of=document,
        document_id=str(document.id),
    )


async def _run(data_chunks, fake_engine, monkeypatch):
    monkeypatch.setattr(task_module, "get_graph_engine", AsyncMock(return_value=fake_engine))
    return await remove_disconnected_chunks(data_chunks)


@pytest.mark.asyncio
async def test_deletes_only_stale_chunks_of_the_given_documents(monkeypatch):
    """Ladybug shape: predecessors are ``{"id": ...}`` dicts.

    The batch holds the document's current chunks; the graph still stores one
    leftover from a previous chunking. Only the leftover is deleted.
    """
    document = _make_document()
    current_a = _make_chunk(document, "first paragraph")
    current_b = _make_chunk(document, "second paragraph")
    stale = _make_chunk(document, "paragraph removed by the rechunk")

    fake_engine = FakeGraphEngine(
        {
            str(document.id): [
                {"id": str(current_a.id)},
                {"id": str(current_b.id)},
                {"id": str(stale.id)},
            ]
        }
    )

    await _run([current_a, current_b], fake_engine, monkeypatch)

    assert fake_engine.predecessor_queries == [(str(document.id), "is_part_of")]
    assert fake_engine.deleted_batches == [[str(stale.id)]]
    assert fake_engine.disconnected_nodes_queries == 0


@pytest.mark.asyncio
async def test_reads_plain_string_predecessors(monkeypatch):
    """Neo4j/Neptune shape: predecessors are bare id strings."""
    document = _make_document()
    current = _make_chunk(document, "kept chunk")
    stale = _make_chunk(document, "stale chunk")

    fake_engine = FakeGraphEngine({str(document.id): [str(current.id), str(stale.id)]})

    await _run([current], fake_engine, monkeypatch)

    assert fake_engine.deleted_batches == [[str(stale.id)]]


@pytest.mark.asyncio
async def test_never_queries_legacy_edges_or_graph_wide_disconnected_nodes(monkeypatch):
    """The old task queried ``has_chunk``/``next_chunk`` (never written by the
    current pipeline) and then deleted every node ``get_disconnected_nodes``
    returned. The fixed task must do neither."""
    document = _make_document()
    current = _make_chunk(document, "kept chunk")

    fake_engine = FakeGraphEngine({str(document.id): [{"id": str(current.id)}]})

    await _run([current], fake_engine, monkeypatch)

    assert fake_engine.successor_queries == []
    assert fake_engine.disconnected_nodes_queries == 0
    assert fake_engine.deleted_batches == []


@pytest.mark.asyncio
async def test_deletes_nothing_when_batch_covers_stored_chunks(monkeypatch):
    document = _make_document()
    current_a = _make_chunk(document, "first paragraph")
    current_b = _make_chunk(document, "second paragraph")

    fake_engine = FakeGraphEngine(
        {str(document.id): [{"id": str(current_a.id)}, {"id": str(current_b.id)}]}
    )

    await _run([current_a, current_b], fake_engine, monkeypatch)

    assert fake_engine.deleted_batches == []


@pytest.mark.asyncio
async def test_deletes_nothing_for_document_with_no_stored_chunks(monkeypatch):
    document = _make_document()
    current = _make_chunk(document, "brand new chunk")

    fake_engine = FakeGraphEngine()

    await _run([current], fake_engine, monkeypatch)

    assert fake_engine.predecessor_queries == [(str(document.id), "is_part_of")]
    assert fake_engine.deleted_batches == []


@pytest.mark.asyncio
async def test_scopes_deletion_to_the_documents_in_the_batch(monkeypatch):
    """Stale chunks of a document outside the batch must survive."""
    batched_document = _make_document("batched.txt")
    other_document = _make_document("other.txt")
    current = _make_chunk(batched_document, "kept chunk")
    other_documents_stale = _make_chunk(other_document, "stale chunk of another document")

    fake_engine = FakeGraphEngine(
        {
            str(batched_document.id): [{"id": str(current.id)}],
            str(other_document.id): [{"id": str(other_documents_stale.id)}],
        }
    )

    await _run([current], fake_engine, monkeypatch)

    assert (str(batched_document.id), "is_part_of") in fake_engine.predecessor_queries
    assert (str(other_document.id), "is_part_of") not in fake_engine.predecessor_queries
    assert fake_engine.deleted_batches == []
