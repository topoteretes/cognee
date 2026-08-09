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


async def _run_build(monkeypatch, session_ids=None, embedding_engine=None):
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
            return_value=embedding_engine or _EmbeddingEngine(),
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
async def test_build_skips_nodes_whose_embedding_batch_failed(monkeypatch):
    """A failed corpus embedding batch must leave those nodes without truth state
    (neutral 1.0 factor), never zero coords at the current epoch (a 0.75x penalty)."""

    class _CorpusFailingEngine(_EmbeddingEngine):
        async def embed_text(self, texts):
            if "alpha corpus" in texts:
                raise RuntimeError("embedding provider down")
            return await super().embed_text(texts)

    result, _vector_engine, graph_engine = await _run_build(
        monkeypatch, embedding_engine=_CorpusFailingEngine()
    )

    assert result["nodes_scored"] == 0
    graph_engine.set_node_truth_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_centroids_commit_after_chunk_scoring(monkeypatch):
    """Centroid upsert (the epoch bump) must happen after chunk-state writes, so a
    mid-build failure leaves the previous epoch fully live."""
    order = []

    result, vector_engine, graph_engine = await _run_build(monkeypatch)

    # Successful build: both writes happened, truth state before centroids.
    assert result["truth_epoch"] == 1
    vector_engine.upsert_raw_vectors.assert_awaited_once()
    graph_engine.set_node_truth_state.assert_awaited_once()

    # mock_calls interleaving across two mocks isn't directly comparable; re-run with
    # side effects that record ordering.
    def record(name):
        async def _side_effect(*args, **kwargs):
            order.append(name)
            return {"chunk-1": True} if name == "truth_state" else None

        return _side_effect

    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    vector_engine = MagicMock()
    vector_engine.retrieve = AsyncMock(return_value=[])
    vector_engine.upsert_raw_vectors = AsyncMock(side_effect=record("centroids"))
    graph_engine = MagicMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(
        return_value=([("learning-1", {"type": "DocumentChunk", "text": "alpha"})], [])
    )
    graph_engine.get_graph_data = AsyncMock(
        return_value=([("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})], [])
    )
    graph_engine.set_node_truth_state = AsyncMock(side_effect=record("truth_state"))
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
        await build_truth_subspace(dataset.id, session_ids=None, user=user)

    assert order == ["truth_state", "centroids"]


@pytest.mark.asyncio
async def test_mid_build_scoring_failure_leaves_old_epoch_live(monkeypatch):
    """If persisting chunk truth states fails, the new centroid epoch must not commit."""
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    vector_engine = MagicMock()
    vector_engine.retrieve = AsyncMock(return_value=[])
    vector_engine.upsert_raw_vectors = AsyncMock()
    graph_engine = MagicMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(
        return_value=([("learning-1", {"type": "DocumentChunk", "text": "alpha"})], [])
    )
    graph_engine.get_graph_data = AsyncMock(
        return_value=([("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})], [])
    )
    graph_engine.set_node_truth_state = AsyncMock(side_effect=RuntimeError("db down"))
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
        result = await build_truth_subspace(dataset.id, session_ids=None, user=user)

    vector_engine.upsert_raw_vectors.assert_not_awaited()
    assert result["truth_epoch"] == 0
    assert result["nodes_scored"] == 0


@pytest.mark.asyncio
async def test_unsupported_backend_skips_before_any_embedding(monkeypatch):
    """A backend inheriting the interface's NotImplementedError default must skip the
    whole build with zero embedding calls."""
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    class _NoTruthBackend:
        set_node_truth_state = GraphDBInterface.set_node_truth_state
        get_nodeset_subgraph = AsyncMock()
        get_graph_data = AsyncMock()

    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    embedding_engine = MagicMock()
    embedding_engine.embed_text = AsyncMock()
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    with (
        patch(
            "cognee.modules.truth_subspace.build.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch(
            "cognee.modules.truth_subspace.build.get_vector_engine_async",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "cognee.modules.truth_subspace.build.get_graph_engine",
            new=AsyncMock(return_value=_NoTruthBackend()),
        ),
        patch(
            "cognee.modules.truth_subspace.build.get_embedding_engine",
            return_value=embedding_engine,
        ),
    ):
        result = await build_truth_subspace(dataset.id, session_ids=None, user=user)

    assert result["skipped"] == "backend_unsupported"
    assert result["nodes_scored"] == 0
    embedding_engine.embed_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unchanged_corpus_rebuild_embeds_no_chunk_texts(monkeypatch):
    """Second build with unchanged learnings + already-scored chunks must not
    embed a single corpus text (only the learning statements re-embed)."""
    from cognee.modules.truth_subspace.centroids import learning_id
    from cognee.modules.truth_subspace.models import TruthCentroidPayload

    embedded_texts = []

    class _TrackingEngine(_EmbeddingEngine):
        async def embed_text(self, texts):
            embedded_texts.extend(texts)
            return await super().embed_text(texts)

    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    existing = [
        TruthCentroidPayload(
            dataset_id=str(dataset.id),
            slot=0,
            count=1,
            truth_epoch=1,
            updated_at=1,
            centroid=[1.0, 0.0],
            learning_ids=[learning_id("alpha")],
        )
    ]
    vector_engine = MagicMock()
    vector_engine.retrieve = AsyncMock(
        return_value=[SimpleNamespace(payload=existing[0].model_dump())]
    )
    vector_engine.upsert_raw_vectors = AsyncMock()

    graph_engine = MagicMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(
        return_value=([("learning-1", {"type": "DocumentChunk", "text": "alpha"})], [])
    )
    graph_engine.get_graph_data = AsyncMock(
        return_value=([("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})], [])
    )
    graph_engine.get_node_truth_state = AsyncMock(
        return_value={"chunk-1": {"truth_alignment": [1.0, 0.0], "truth_epoch": 1}}
    )
    graph_engine.set_node_truth_state = AsyncMock()
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
            return_value=_TrackingEngine(),
        ),
    ):
        # session_ids -> extend mode, so the pre-existing centroid slot is reused.
        result = await build_truth_subspace(dataset.id, session_ids=["s-1"], user=user)

    assert result["truth_epoch"] == 1
    assert result["nodes_scored"] == 0
    assert "alpha corpus" not in embedded_texts  # corpus untouched
    graph_engine.set_node_truth_state.assert_not_awaited()
    vector_engine.upsert_raw_vectors.assert_not_awaited()  # centroids unchanged


@pytest.mark.asyncio
async def test_new_chunks_reuse_stored_vectors_instead_of_embedding(monkeypatch):
    """Chunk vectors already stored in DocumentChunk_text are read back; only
    texts the store cannot return get embedded."""
    embedded_texts = []

    class _TrackingEngine(_EmbeddingEngine):
        async def embed_text(self, texts):
            embedded_texts.extend(texts)
            return await super().embed_text(texts)

    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    vector_engine = MagicMock()

    async def retrieve(collection_name, ids, include_vector=False):
        if collection_name == "DocumentChunk_text" and include_vector:
            return [SimpleNamespace(id="chunk-1", payload={"vector": [1.0, 0.0]})]
        return []

    vector_engine.retrieve = AsyncMock(side_effect=retrieve)
    vector_engine.upsert_raw_vectors = AsyncMock()

    graph_engine = MagicMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(
        return_value=([("learning-1", {"type": "DocumentChunk", "text": "alpha"})], [])
    )
    graph_engine.get_graph_data = AsyncMock(
        return_value=([("chunk-1", {"type": "DocumentChunk", "text": "alpha corpus"})], [])
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
            return_value=_TrackingEngine(),
        ),
    ):
        result = await build_truth_subspace(dataset.id, session_ids=None, user=user)

    assert result["nodes_scored"] == 1
    assert embedded_texts == ["alpha"]  # learning statement only; corpus reused


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
