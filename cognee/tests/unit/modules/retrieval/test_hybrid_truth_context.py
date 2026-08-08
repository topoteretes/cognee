from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.truth_subspace.models import TruthCentroidPayload


def _unified_engine():
    engine = MagicMock()
    engine.vector = MagicMock()
    engine.graph = MagicMock()
    engine.graph.get_node_truth_state = AsyncMock(
        return_value={"chunk-1": {"truth_alignment": [1.0], "truth_epoch": 3}}
    )
    return engine


@pytest.mark.asyncio
async def test_truth_context_loads_exact_centroid_slots_for_current_dataset():
    retriever = HybridRetriever(chunks_top_k=1, use_truth_weight=True)
    retriever._unified_engine = _unified_engine()
    centroid_loader = AsyncMock(
        return_value=[
            TruthCentroidPayload(
                dataset_id="dataset-1",
                slot=0,
                count=1,
                truth_epoch=3,
                updated_at=123,
                centroid=[1.0, 0.0, 0.0],
            )
        ]
    )
    token = current_dataset_id.set("dataset-1")

    try:
        with (
            patch(
                "cognee.modules.retrieval.hybrid_retriever.load_centroids",
                centroid_loader,
            ),
            patch.object(
                retriever,
                "_candidate_chunk_ids",
                new=AsyncMock(return_value=["chunk-1"]),
            ),
        ):
            q_coords, truth_state_by_id, current_truth_epoch = await retriever._build_truth_context(
                [1.0, 0.0, 0.0]
            )
    finally:
        current_dataset_id.reset(token)

    centroid_loader.assert_awaited_once_with(retriever._unified_engine.vector, "dataset-1")
    retriever._unified_engine.graph.get_node_truth_state.assert_awaited_once_with(["chunk-1"])
    assert q_coords == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert truth_state_by_id == {"chunk-1": {"truth_alignment": [1.0], "truth_epoch": 3}}
    assert current_truth_epoch == 3
