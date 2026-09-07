"""Plan item B4 for the truth-subspace build.

Three guarantees, each pinned here:

* the backend is probed for ``set_node_truth_state`` before any embedding call, and an
  unsupported adapter yields ``skipped: backend_unsupported`` with zero embeddings;
* a chunk whose embedding batch failed is skipped, never persisted as all-zero coordinates;
* chunk coordinates are persisted at epoch N+1 first and the N+1 centroids are upserted
  last, so a failure anywhere before that final write leaves epoch N live.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.modules.truth_subspace.build as build_module
from cognee.modules.truth_subspace.build import (
    REASON_BACKEND_UNSUPPORTED,
    STATUS_COMPLETED,
    STATUS_ERRORED,
    STATUS_SKIPPED,
    build_truth_subspace,
)

LEARNINGS = [
    ("learning-1", {"type": "DocumentChunk", "text": "alpha"}),
    ("learning-2", {"type": "DocumentChunk", "text": "beta"}),
]
VECTORS = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "alpha corpus": [1.0, 0.0],
    "beta corpus": [0.0, 1.0],
}


class RecordingEmbeddingEngine:
    """Deterministic embeddings; raises for any batch containing a text in ``fail_on``."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = []

    async def embed_text(self, texts):
        self.calls.append(list(texts))
        if self.fail_on.intersection(texts):
            raise RuntimeError("embedding provider down")
        return [VECTORS[text] for text in texts]


def _graph_engine(chunks, *, supports_truth_state=None):
    graph_engine = MagicMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(return_value=(LEARNINGS, []))
    graph_engine.get_graph_data = AsyncMock(return_value=(chunks, []))
    graph_engine.set_node_truth_state = AsyncMock(
        side_effect=lambda scored: {node_id: True for node_id in scored}
    )
    if supports_truth_state is not None:
        graph_engine.supports_truth_state = supports_truth_state
    return graph_engine


async def _run_build(
    monkeypatch,
    *,
    graph_engine,
    embedding_engine,
    vector_engine=None,
    session_ids=None,
):
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    if vector_engine is None:
        vector_engine = MagicMock()
        vector_engine.retrieve = AsyncMock(return_value=[])
        vector_engine.upsert_raw_vectors = AsyncMock()
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with (
        patch.object(
            build_module,
            "get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch.object(
            build_module, "get_vector_engine_async", new=AsyncMock(return_value=vector_engine)
        ),
        patch.object(build_module, "get_graph_engine", new=AsyncMock(return_value=graph_engine)),
        patch.object(build_module, "get_embedding_engine", return_value=embedding_engine),
    ):
        result = await build_truth_subspace(dataset.id, session_ids=session_ids, user=user)
    return result, vector_engine


@pytest.mark.asyncio
async def test_unsupported_backend_is_skipped_before_any_embedding_call(monkeypatch):
    graph_engine = _graph_engine(
        [("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})],
        supports_truth_state=False,
    )
    embedding_engine = RecordingEmbeddingEngine()

    result, vector_engine = await _run_build(
        monkeypatch, graph_engine=graph_engine, embedding_engine=embedding_engine
    )

    assert result["status"] == STATUS_SKIPPED
    assert result["reason"] == REASON_BACKEND_UNSUPPORTED
    assert result["anchors"] == 0
    assert result["nodes_scored"] == 0
    assert result["truth_epoch"] == 0
    # Zero embedding calls, zero reads of the learnings, zero writes.
    assert embedding_engine.calls == []
    graph_engine.get_nodeset_subgraph.assert_not_awaited()
    graph_engine.get_graph_data.assert_not_awaited()
    graph_engine.set_node_truth_state.assert_not_awaited()
    vector_engine.upsert_raw_vectors.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunk_coordinates_are_written_before_the_centroids(monkeypatch):
    """Epoch N+1 chunk state lands first; the N+1 centroids are the final write."""
    order = []
    graph_engine = _graph_engine(
        [
            ("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"}),
            ("chunk-2", {"type": "DocumentChunk", "text": "beta corpus"}),
        ]
    )

    async def _set_truth_state(scored):
        order.append("set_node_truth_state")
        return {node_id: True for node_id in scored}

    graph_engine.set_node_truth_state = AsyncMock(side_effect=_set_truth_state)
    vector_engine = MagicMock()
    vector_engine.retrieve = AsyncMock(return_value=[])
    vector_engine.upsert_raw_vectors = AsyncMock(
        side_effect=lambda *args, **kwargs: order.append("upsert_centroids")
    )

    result, _ = await _run_build(
        monkeypatch,
        graph_engine=graph_engine,
        embedding_engine=RecordingEmbeddingEngine(),
        vector_engine=vector_engine,
    )

    assert order == ["set_node_truth_state", "upsert_centroids"]
    assert result["status"] == STATUS_COMPLETED
    assert result["truth_epoch"] == 1
    assert result["anchors"] == 2
    assert result["nodes_scored"] == 2
    assert result["nodes_skipped"] == 0

    scored = graph_engine.set_node_truth_state.await_args.args[0]
    assert {state["truth_epoch"] for state in scored.values()} == {1}
    points = vector_engine.upsert_raw_vectors.await_args.args[1]
    assert {point["payload"]["truth_epoch"] for point in points} == {1}


@pytest.mark.asyncio
async def test_failed_embedding_batch_skips_those_chunks_instead_of_zero_filling(monkeypatch):
    # One text per batch so a single failing batch maps to a single chunk.
    monkeypatch.setattr(build_module, "NODE_EMBED_BATCH_SIZE", 1)
    graph_engine = _graph_engine(
        [
            ("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"}),
            ("chunk-bad", {"type": "DocumentChunk", "text": "unembeddable"}),
            ("chunk-2", {"type": "DocumentChunk", "text": "beta corpus"}),
        ]
    )

    result, vector_engine = await _run_build(
        monkeypatch,
        graph_engine=graph_engine,
        embedding_engine=RecordingEmbeddingEngine(fail_on={"unembeddable"}),
    )

    scored = graph_engine.set_node_truth_state.await_args.args[0]
    assert set(scored) == {"chunk-1", "chunk-2"}
    assert "chunk-bad" not in scored
    for state in scored.values():
        # Real coordinates for the survivors — never an all-zero "neutral" row.
        assert any(value != 0.0 for value in state["truth_alignment"])
    assert result["status"] == STATUS_COMPLETED
    assert result["nodes_scored"] == 2
    assert result["nodes_skipped"] == 1
    assert result["truth_epoch"] == 1
    vector_engine.upsert_raw_vectors.assert_awaited_once()


@pytest.mark.asyncio
async def test_all_embeddings_failing_keeps_epoch_n_live(monkeypatch):
    graph_engine = _graph_engine(
        [("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})],
    )

    result, vector_engine = await _run_build(
        monkeypatch,
        graph_engine=graph_engine,
        embedding_engine=RecordingEmbeddingEngine(fail_on={"alpha corpus"}),
    )

    assert result["status"] == STATUS_ERRORED
    assert result["truth_epoch"] == 0
    assert result["anchors"] == 0
    assert result["nodes_scored"] == 0
    assert result["nodes_skipped"] == 1
    graph_engine.set_node_truth_state.assert_not_awaited()
    vector_engine.upsert_raw_vectors.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_chunk_write_leaves_epoch_n_live(monkeypatch):
    """set_node_truth_state raising means the N+1 centroids never become live."""
    graph_engine = _graph_engine(
        [("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})],
    )
    graph_engine.set_node_truth_state = AsyncMock(side_effect=RuntimeError("graph write failed"))

    result, vector_engine = await _run_build(
        monkeypatch, graph_engine=graph_engine, embedding_engine=RecordingEmbeddingEngine()
    )

    assert result["status"] == STATUS_ERRORED
    assert "graph write failed" in result["error"]
    assert result["truth_epoch"] == 0
    assert result["anchors"] == 0
    assert result["nodes_scored"] == 0
    vector_engine.upsert_raw_vectors.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_node_load_leaves_epoch_n_live(monkeypatch):
    graph_engine = _graph_engine([])
    graph_engine.get_graph_data = AsyncMock(side_effect=RuntimeError("graph down"))
    embedding_engine = RecordingEmbeddingEngine()

    result, vector_engine = await _run_build(
        monkeypatch, graph_engine=graph_engine, embedding_engine=embedding_engine
    )

    assert result["status"] == STATUS_ERRORED
    assert result["truth_epoch"] == 0
    vector_engine.upsert_raw_vectors.assert_not_awaited()
    graph_engine.set_node_truth_state.assert_not_awaited()
    # Only the learnings were embedded (in learning-id order); no chunk embedding ran.
    assert [sorted(call) for call in embedding_engine.calls] == [["alpha", "beta"]]


@pytest.mark.asyncio
async def test_no_scoreable_chunks_still_commits_the_new_centroids(monkeypatch):
    """With nothing to score, nothing can be left inconsistent: centroids go live."""
    graph_engine = _graph_engine([("entity-1", {"type": "Entity", "name": "not a chunk"})])

    result, vector_engine = await _run_build(
        monkeypatch, graph_engine=graph_engine, embedding_engine=RecordingEmbeddingEngine()
    )

    assert result["status"] == STATUS_COMPLETED
    assert result["truth_epoch"] == 1
    assert result["anchors"] == 2
    assert result["nodes_scored"] == 0
    vector_engine.upsert_raw_vectors.assert_awaited_once()
    graph_engine.set_node_truth_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_centroid_commit_reports_the_previous_epoch(monkeypatch):
    graph_engine = _graph_engine(
        [("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})],
    )
    vector_engine = MagicMock()
    vector_engine.retrieve = AsyncMock(return_value=[])
    vector_engine.upsert_raw_vectors = AsyncMock(side_effect=RuntimeError("vector store down"))

    result, _ = await _run_build(
        monkeypatch,
        graph_engine=graph_engine,
        embedding_engine=RecordingEmbeddingEngine(),
        vector_engine=vector_engine,
    )

    # Chunks were scored at N+1, but the commit failed, so N (here: nothing) stays live.
    graph_engine.set_node_truth_state.assert_awaited_once()
    assert result["status"] == STATUS_ERRORED
    assert "vector store down" in result["error"]
    assert result["truth_epoch"] == 0
    assert result["anchors"] == 0
    assert result["nodes_scored"] == 1
