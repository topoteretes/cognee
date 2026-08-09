from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks
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
        with patch(
            "cognee.modules.retrieval.hybrid_retriever.load_centroids",
            centroid_loader,
        ):
            q_coords, current_truth_epoch = await retriever._build_truth_context([1.0, 0.0, 0.0])
    finally:
        current_dataset_id.reset(token)

    centroid_loader.assert_awaited_once_with(retriever._unified_engine.vector, "dataset-1")
    assert q_coords == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert current_truth_epoch == 3
    # No duplicate DocumentChunk_text search: truth states are fetched inside the
    # chunk lane for its own candidates, not here.
    retriever._unified_engine.graph.get_node_truth_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_truth_context_off_returns_baseline():
    retriever = HybridRetriever(chunks_top_k=1, use_truth_weight=False)
    retriever._unified_engine = _unified_engine()

    assert await retriever._build_truth_context([1.0]) == (None, None)
    assert retriever._truth_state_fetcher(None, None) is None


def test_truth_state_fetcher_gated_on_context():
    retriever = HybridRetriever(chunks_top_k=1, use_truth_weight=True)
    retriever._unified_engine = _unified_engine()

    fetcher = retriever._truth_state_fetcher([1.0, 0.0], 3)
    assert fetcher is retriever._unified_engine.graph.get_node_truth_state
    assert retriever._truth_state_fetcher(None, 3) is None
    assert retriever._truth_state_fetcher([1.0], None) is None


def _chunk_payload(chunk_id: str):
    return {"id": chunk_id, "text": f"text {chunk_id}", "payload": {"id": chunk_id}}


class _VectorResult:
    def __init__(self, chunk_id):
        self.id = chunk_id
        self.payload = _chunk_payload(chunk_id)
        self.score = 0.5


@pytest.mark.asyncio
async def test_chunk_lane_fetches_truth_state_once_and_reranks():
    """End to end through the chunk lane: exactly one DocumentChunk_text search,
    one batched truth-state fetch over the lane's own candidates, and the
    aligned chunk outranks the misaligned one that tied on retrieval rank."""
    vector_engine = MagicMock()

    async def search(collection_name, *args, **kwargs):
        if collection_name == "DocumentChunk_text":
            return [_VectorResult("aligned"), _VectorResult("misaligned")]
        return []

    vector_engine.search = AsyncMock(side_effect=search)
    vector_engine.retrieve = AsyncMock(return_value=[])

    fetch_truth_state = AsyncMock(
        return_value={
            "aligned": {"truth_alignment": [1.0] + [0.0] * 7, "truth_epoch": 3},
            "misaligned": {"truth_alignment": [0.0] * 8, "truth_epoch": 3},
        }
    )

    with patch(
        "cognee.modules.retrieval.hybrid.chunks.search_bm25_chunks",
        new=AsyncMock(return_value=[]),
    ):
        result = await retrieve_hybrid_chunks(
            vector_engine=vector_engine,
            query="q",
            chunks_top_k=2,
            text_summaries_top_k=0,
            node_name=None,
            node_name_filter_operator="OR",
            use_importance_weight=False,
            query_vector=[1.0, 0.0],
            use_truth_weight=True,
            q_coords=[1.0] + [0.0] * 7,
            current_truth_epoch=3,
            fetch_truth_state=fetch_truth_state,
        )

    fetch_truth_state.assert_awaited_once()
    assert set(fetch_truth_state.await_args.args[0]) == {"aligned", "misaligned"}
    chunk_searches = [
        call
        for call in vector_engine.search.await_args_list
        if call.args[0] == "DocumentChunk_text"
    ]
    assert len(chunk_searches) == 1
    ranked_ids = [getattr(chunk, "id", None) for chunk in result["chunks"]]
    assert ranked_ids == ["aligned", "misaligned"]
