from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.hybrid.truth import TruthContext
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.truth_subspace.models import TruthCentroidPayload


def _result(result_id="chunk-1", payload=None):
    scored_result = MagicMock()
    scored_result.id = result_id
    scored_result.payload = payload or {"id": result_id, "text": result_id}
    return scored_result


def _unified_engine():
    engine = MagicMock()
    engine.vector = MagicMock()
    engine.vector.embedding_engine.embed_text = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    engine.vector.search = AsyncMock(return_value=[_result()])
    engine.graph = MagicMock()
    engine.graph.is_empty = AsyncMock(return_value=False)
    engine.graph.get_neighborhood = AsyncMock(return_value=([], []))
    engine.graph.get_node_truth_state = AsyncMock(
        return_value={"chunk-1": {"truth_alignment": [1.0], "truth_epoch": 3}}
    )
    return engine


@pytest.mark.asyncio
async def test_truth_context_loads_exact_centroid_slots_for_current_dataset():
    retriever = HybridRetriever(chunks_top_k=1, use_truth_weight=True)
    engine = _unified_engine()
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
                "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ),
            patch(
                "cognee.modules.retrieval.hybrid.truth.load_centroids",
                centroid_loader,
            ),
        ):
            retrieved = await retriever.get_retrieved_objects(query="q")
    finally:
        current_dataset_id.reset(token)

    centroid_loader.assert_awaited_once_with(engine.vector, "dataset-1")
    engine.graph.get_node_truth_state.assert_awaited_once_with(["chunk-1"])
    assert retrieved["chunks"][0].id == "chunk-1"


@pytest.mark.asyncio
async def test_retriever_threads_truth_context_into_chunk_lane():
    retriever = HybridRetriever(use_truth_weight=True)
    engine = _unified_engine()
    truth = TruthContext(
        q_coords=[1.0],
        truth_state_by_id={"chunk-1": {"truth_epoch": 3}},
        current_truth_epoch=3,
    )

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=engine,
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.build_truth_context",
            new_callable=AsyncMock,
            return_value=truth,
        ) as build,
        patch(
            "cognee.modules.retrieval.hybrid_retriever.retrieve_hybrid_chunks",
            new_callable=AsyncMock,
            return_value={"chunks": [], "chunk_summaries": {}},
        ) as retrieve_chunks,
    ):
        await retriever.get_retrieved_objects(query="q")

    build.assert_awaited_once()
    assert retrieve_chunks.await_args.kwargs["q_coords"] == truth.q_coords
    assert retrieve_chunks.await_args.kwargs["truth_state_by_id"] == truth.truth_state_by_id
    assert retrieve_chunks.await_args.kwargs["current_truth_epoch"] == truth.current_truth_epoch
