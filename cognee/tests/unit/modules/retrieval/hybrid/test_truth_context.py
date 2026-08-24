from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.hybrid.truth import TruthContext, build_truth_context
from cognee.modules.truth_subspace.models import TruthCentroidPayload


QUERY_VECTOR = [1.0, 0.0, 0.0]


def _engine():
    engine = MagicMock()
    engine.vector = MagicMock()
    engine.graph = MagicMock()
    engine.graph.get_node_truth_state = AsyncMock(return_value={"chunk-1": {"truth_epoch": 3}})
    return engine


def _centroid(epoch=3):
    return TruthCentroidPayload(
        dataset_id="dataset-1",
        slot=0,
        count=1,
        truth_epoch=epoch,
        updated_at=123,
        centroid=[1.0, 0.0, 0.0],
    )


def _hit(chunk_id="chunk-1"):
    hit = MagicMock()
    hit.id = chunk_id
    hit.payload = {"id": chunk_id}
    return hit


async def _build(*, use_truth_weight=True, engine=None, node_name=None):
    return await build_truth_context(
        engine if engine is not None else _engine(),
        QUERY_VECTOR,
        use_truth_weight=use_truth_weight,
        chunks_top_k=1,
        node_name=node_name,
        node_name_filter_operator="OR",
    )


@pytest.mark.asyncio
async def test_truth_off_makes_no_engine_calls():
    engine = _engine()

    with patch(
        "cognee.modules.retrieval.hybrid.truth.load_centroids", new_callable=AsyncMock
    ) as load:
        truth = await _build(use_truth_weight=False, engine=engine)

    assert truth == TruthContext()
    load.assert_not_awaited()
    engine.vector.search.assert_not_called()
    engine.graph.get_node_truth_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_dataset_id_returns_empty_context():
    engine = _engine()
    token = current_dataset_id.set(None)
    try:
        with patch(
            "cognee.modules.retrieval.hybrid.truth.load_centroids", new_callable=AsyncMock
        ) as load:
            truth = await _build(engine=engine)
    finally:
        current_dataset_id.reset(token)

    assert truth == TruthContext()
    load.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_centroids_returns_empty_context():
    engine = _engine()
    token = current_dataset_id.set("dataset-1")
    try:
        with patch(
            "cognee.modules.retrieval.hybrid.truth.load_centroids",
            new_callable=AsyncMock,
            return_value=[],
        ) as load:
            truth = await _build(engine=engine)
    finally:
        current_dataset_id.reset(token)

    assert truth == TruthContext()
    load.assert_awaited_once_with(engine.vector, "dataset-1")
    engine.graph.get_node_truth_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_centroids_populate_coords_and_epoch():
    engine = _engine()
    token = current_dataset_id.set("dataset-1")
    try:
        with (
            patch(
                "cognee.modules.retrieval.hybrid.truth.load_centroids",
                new_callable=AsyncMock,
                return_value=[_centroid(epoch=3)],
            ),
            patch(
                "cognee.modules.retrieval.hybrid.truth.search_collection",
                new_callable=AsyncMock,
                return_value=[_hit()],
            ),
        ):
            truth = await _build(engine=engine)
    finally:
        current_dataset_id.reset(token)

    assert truth.q_coords == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert truth.current_truth_epoch == 3
    assert truth.truth_state_by_id == {"chunk-1": {"truth_epoch": 3}}
    engine.graph.get_node_truth_state.assert_awaited_once_with(["chunk-1"])


@pytest.mark.asyncio
async def test_load_centroids_error_fails_open(caplog):
    engine = _engine()
    token = current_dataset_id.set("dataset-1")
    try:
        with (
            caplog.at_level("DEBUG", logger="HybridRetriever"),
            patch(
                "cognee.modules.retrieval.hybrid.truth.load_centroids",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            truth = await _build(engine=engine)
    finally:
        current_dataset_id.reset(token)

    assert truth == TruthContext()
    assert "Truth-subspace lookup failed" in caplog.text
