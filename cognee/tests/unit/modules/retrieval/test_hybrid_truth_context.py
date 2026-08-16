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
        with patch(
            "cognee.modules.retrieval.hybrid_retriever.load_centroids",
            centroid_loader,
        ):
            q_coords, current_truth_epoch = await retriever._build_truth_context(
                [1.0, 0.0, 0.0]
            )
    finally:
        current_dataset_id.reset(token)

    centroid_loader.assert_awaited_once_with(retriever._unified_engine.vector, "dataset-1")
    assert q_coords == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert current_truth_epoch == 3


@pytest.mark.asyncio
async def test_retrieve_hybrid_chunks_queries_graph_for_all_candidates():
    from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks

    bm25_chunk = {"id": "bm25-1", "text": "BM25 match"}
    vector_chunk = {"id": "vector-1", "text": "Vector match"}
    summary_chunk = {"id": "summary-1", "text": "Summary match", "source_chunk_id": "source-1"}

    mock_bm25 = AsyncMock(return_value=[bm25_chunk])
    mock_search = AsyncMock(side_effect=lambda engine, coll, *args, **kwargs: (
        [vector_chunk] if coll == "DocumentChunk_text" else [summary_chunk]
    ))
    mock_load_source = AsyncMock(return_value=[{"id": "source-1", "text": "Source match"}])

    mock_graph = MagicMock()
    mock_graph.get_node_truth_state = AsyncMock(return_value={
        "bm25-1": {"truth_alignment": [1.0], "truth_epoch": 2},
        "vector-1": {"truth_alignment": [1.0], "truth_epoch": 2},
        "source-1": {"truth_alignment": [1.0], "truth_epoch": 2},
    })

    with (
        patch("cognee.modules.retrieval.hybrid.chunks.search_bm25_chunks", mock_bm25),
        patch("cognee.modules.retrieval.hybrid.chunks.search_collection", mock_search),
        patch("cognee.modules.retrieval.hybrid.chunks.load_source_chunks_for_summaries", mock_load_source),
    ):
        result = await retrieve_hybrid_chunks(
            vector_engine=MagicMock(),
            query="test query",
            chunks_top_k=2,
            text_summaries_top_k=1,
            node_name=None,
            node_name_filter_operator="OR",
            use_importance_weight=False,
            query_vector=[1.0],
            use_truth_weight=True,
            q_coords=[1.0],
            current_truth_epoch=2,
            graph_engine=mock_graph,
        )

    called_ids = mock_graph.get_node_truth_state.call_args[0][0]
    assert set(called_ids) == {"bm25-1", "vector-1", "source-1"}

