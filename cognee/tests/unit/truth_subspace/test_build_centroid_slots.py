from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.truth_subspace.build import build_truth_subspace


class _EmbeddingEngine:
    async def embed_text(self, texts):
        vectors_by_text = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "alpha corpus": [1.0, 0.0],
        }
        return [vectors_by_text[text] for text in texts]


async def _run_build(monkeypatch, session_ids=None):
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    vector_engine = MagicMock()
    vector_engine.retrieve = AsyncMock(return_value=[])
    vector_engine.upsert_raw_vectors = AsyncMock()

    graph_engine = MagicMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(
        return_value=(
            [
                ("learning-1", {"type": "DocumentChunk", "text": "alpha"}),
                ("learning-2", {"type": "DocumentChunk", "text": "beta"}),
            ],
            [],
        )
    )
    graph_engine.get_graph_data = AsyncMock(
        return_value=(
            [
                ("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"}),
            ],
            [],
        )
    )
    graph_engine.set_node_truth_state = AsyncMock(return_value={"chunk-1": True})
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with (
        patch(
            "cognee.modules.truth_subspace.build.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch(
            "cognee.modules.truth_subspace.build.get_vector_engine_async",
            new=AsyncMock(return_value=vector_engine),
        ),
        patch(
            "cognee.modules.truth_subspace.build.get_graph_engine",
            new=AsyncMock(return_value=graph_engine),
        ),
        patch(
            "cognee.modules.truth_subspace.build.get_embedding_engine",
            return_value=_EmbeddingEngine(),
        ),
    ):
        result = await build_truth_subspace(dataset.id, session_ids=session_ids, user=user)

    return result, vector_engine, graph_engine


@pytest.mark.asyncio
async def test_build_truth_subspace_writes_centroids_and_epoch_state(monkeypatch):
    result, vector_engine, graph_engine = await _run_build(monkeypatch)

    assert result["anchors"] == 2
    assert result["nodes_scored"] == 1
    assert result["truth_epoch"] == 1
    vector_engine.upsert_raw_vectors.assert_awaited_once()
    graph_engine.set_node_truth_state.assert_awaited_once()

    node_state = graph_engine.set_node_truth_state.await_args.args[0]
    assert node_state["chunk-1"]["truth_epoch"] == 1
    assert len(node_state["chunk-1"]["truth_alignment"]) == 8
    assert sorted(node_state["chunk-1"]["truth_alignment"][:2]) == [0.0, 1.0]
    assert node_state["chunk-1"]["truth_alignment"][2:] == [0.0] * 6


@pytest.mark.asyncio
async def test_build_truth_subspace_filters_learning_sets_by_session_ids(monkeypatch):
    _result, _vector_engine, graph_engine = await _run_build(
        monkeypatch, session_ids=["s-1", "s-2"]
    )

    graph_engine.get_nodeset_subgraph.assert_awaited_once()
    assert graph_engine.get_nodeset_subgraph.await_args.kwargs["node_name"] == [
        "session_learnings:s-1",
        "session_learnings:s-2",
    ]
