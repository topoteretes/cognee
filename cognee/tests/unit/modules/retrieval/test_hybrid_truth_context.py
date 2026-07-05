from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.truth_subspace.models import TruthCentroidPayload

CURRENT_EPOCH = 7


def _unified_engine():
    engine = MagicMock()
    engine.vector = MagicMock()
    engine.graph = MagicMock()
    engine.graph.get_node_truth_state = AsyncMock(
        return_value={"chunk-1": {"truth_alignment": [1.0], "truth_epoch": 3}}
    )
    return engine


def _result(result_id="result-id", payload=None):
    scored_result = MagicMock()
    scored_result.id = result_id
    scored_result.payload = payload
    return scored_result


def _chunk_id(chunk):
    if isinstance(chunk, dict):
        return chunk.get("id")
    return chunk.payload.get("id")


@pytest.mark.asyncio
async def test_truth_context_returns_query_coords_without_prefetching_truth_state():
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
        with patch(
            "cognee.modules.retrieval.hybrid_retriever.load_centroids",
            centroid_loader,
        ):
            q_coords, current_truth_epoch = await retriever._build_truth_context([1.0, 0.0, 0.0])
    finally:
        current_dataset_id.reset(token)

    centroid_loader.assert_awaited_once_with(retriever._unified_engine.vector, "dataset-1")
    # The truth-state map is now built from the full candidate set inside the chunk
    # lane, so the context builder must not pre-fetch it from a vector-only window.
    retriever._unified_engine.graph.get_node_truth_state.assert_not_awaited()
    assert q_coords == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert current_truth_epoch == 3


@pytest.mark.asyncio
async def test_truth_state_covers_bm25_and_summary_only_candidates():
    # Regression for #3653: chunks surfaced only via BM25 or the summary channel are
    # absent from the vector candidate window. The truth-state map must still cover
    # them so every channel gets a truth factor instead of a neutral 1.0.
    vector = MagicMock()

    async def search(collection_name, *args, **kwargs):
        if collection_name == "DocumentChunk_text":
            return [_result("vector-1", {"id": "vector-1", "text": "Vector chunk"})]
        if collection_name == "TextSummary_text":
            return [
                _result(
                    "summary-1",
                    {"id": "summary-1", "text": "Summary", "source_chunk_id": "summary-only"},
                )
            ]
        return []

    async def retrieve(collection_name, ids):
        if collection_name == "DocumentChunk_text":
            return [_result("summary-only", {"id": "summary-only", "text": "Summary source"})]
        return []

    vector.search = AsyncMock(side_effect=search)
    vector.retrieve = AsyncMock(side_effect=retrieve)

    graph = MagicMock()
    graph.get_node_truth_state = AsyncMock(
        return_value={
            "vector-1": {"truth_alignment": [0.5], "truth_epoch": CURRENT_EPOCH},  # factor 1.0
            "bm25-only": {"truth_alignment": [0.0], "truth_epoch": CURRENT_EPOCH},  # factor 0.75
            "summary-only": {"truth_alignment": [0.0], "truth_epoch": CURRENT_EPOCH},  # factor 0.75
        }
    )

    bm25_retriever = MagicMock()
    bm25_retriever.get_retrieved_objects = AsyncMock(
        return_value=[({"id": "bm25-only", "text": "BM25 chunk"}, 2.0)]
    )

    with patch(
        "cognee.modules.retrieval.hybrid.chunks.BM25ChunksRetriever",
        return_value=bm25_retriever,
    ):
        result = await retrieve_hybrid_chunks(
            vector_engine=vector,
            graph_engine=graph,
            query="q",
            chunks_top_k=3,
            text_summaries_top_k=None,
            node_name=None,
            node_name_filter_operator="OR",
            use_importance_weight=False,
            query_vector=[0.1, 0.2],
            use_truth_weight=True,
            q_coords=[1.0],
            current_truth_epoch=CURRENT_EPOCH,
        )

    # The truth map is fetched over the full assembled candidate set, not just the
    # vector window: BM25-only and summary-only ids must be included.
    graph.get_node_truth_state.assert_awaited_once()
    requested_ids = set(graph.get_node_truth_state.await_args.args[0])
    assert {"vector-1", "bm25-only", "summary-only"} <= requested_ids

    # The 0.75 truth factor is actually applied to the BM25-only and summary-only
    # chunks, demoting them below the neutral (1.0) vector chunk. Without the fix
    # they would keep a neutral 1.0 and tie, leaving vector-1 last by id order.
    assert [_chunk_id(chunk) for chunk in result["chunks"]] == [
        "vector-1",
        "bm25-only",
        "summary-only",
    ]
