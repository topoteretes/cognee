from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


QUERY_VECTOR = [0.1, 0.2, 0.3]


def _result(result_id, payload):
    scored_result = MagicMock()
    scored_result.id = result_id
    scored_result.payload = payload
    return scored_result


def _unified(vector=None):
    unified = MagicMock()
    unified.vector = vector or MagicMock()
    unified.vector.embedding_engine.embed_text = AsyncMock(return_value=[QUERY_VECTOR])
    unified.graph = MagicMock()
    unified.graph.is_empty = AsyncMock(return_value=False)
    unified.graph.get_neighborhood = AsyncMock(return_value=([], []))
    return unified


@pytest.mark.asyncio
async def test_query_batch_returns_aligned_results_contexts_and_completions():
    chunk_calls = {"n": 0}

    async def search(collection_name, *args, **kwargs):
        if collection_name != "DocumentChunk_text":
            return []
        chunk_calls["n"] += 1
        chunk_id = f"c{chunk_calls['n']}"
        return [_result(chunk_id, {"id": chunk_id, "text": f"Chunk {chunk_calls['n']}"})]

    vector = MagicMock()
    vector.search = AsyncMock(side_effect=search)
    retriever = HybridRetriever(text_summaries_top_k=0)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(vector=vector),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.generate_completion_batch",
            new_callable=AsyncMock,
            return_value=["answer-1", "answer-2"],
        ) as complete_batch,
    ):
        answers = await retriever.get_completion(query_batch=["q1", "q2"])

    assert answers == ["answer-1", "answer-2"]
    assert complete_batch.await_args.kwargs["query_batch"] == ["q1", "q2"]
    assert complete_batch.await_args.kwargs["context"] == [
        "## Relevant passages\nChunk 1",
        "## Relevant passages\nChunk 2",
    ]


@pytest.mark.asyncio
async def test_empty_graph_query_batch_returns_empty_shapes():
    unified = _unified()
    unified.graph.is_empty = AsyncMock(return_value=True)
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=unified,
    ):
        retrieved = await retriever.get_retrieved_objects(query_batch=["q1", "q2"])

    assert retrieved == [empty_hybrid_result(), empty_hybrid_result()]
    unified.vector.embedding_engine.embed_text.assert_not_awaited()
