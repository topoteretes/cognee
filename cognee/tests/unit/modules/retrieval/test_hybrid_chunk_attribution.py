from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks, search_bm25_chunks


def _result(result_id: str, text: str, score: float, **payload):
    return SimpleNamespace(
        id=result_id,
        score=score,
        payload={"id": result_id, "text": text, **payload},
    )


def _bm25(results):
    return SimpleNamespace(
        cache_status="test",
        get_retrieved_objects=AsyncMock(return_value=results),
    )


async def _retrieve(vector, bm25):
    with patch(
        "cognee.modules.retrieval.hybrid.chunks.BM25ChunksRetriever",
        return_value=bm25,
    ):
        return await retrieve_hybrid_chunks(
            vector_engine=vector,
            query="project status",
            chunks_top_k=2,
            text_summaries_top_k=2,
            node_name=None,
            node_name_filter_operator="OR",
            use_importance_weight=False,
            query_vector=[0.1, 0.2],
        )


@pytest.mark.asyncio
async def test_native_channel_scores_survive_pairing_ranking_and_result_projection():
    chunk_id = str(uuid4())
    summary_id = str(uuid4())
    bm25 = _bm25([({"id": chunk_id, "text": "Shared passage"}, 8.25)])
    vector = MagicMock()

    async def search(collection_name, *_args, **_kwargs):
        if collection_name == "DocumentChunk_text":
            return [_result(chunk_id, "Shared passage", 0.12)]
        if collection_name == "TextSummary_text":
            return [
                _result(
                    summary_id,
                    "Passage summary",
                    0.21,
                    source_chunk_id=chunk_id,
                )
            ]
        return []

    vector.search = AsyncMock(side_effect=search)
    vector.retrieve = AsyncMock(return_value=[_result(chunk_id, "Shared passage", 0.0)])

    result = await _retrieve(vector, bm25)

    attribution = result["chunk_attribution"][0]
    scores = {channel["channel"]: channel["native_score"] for channel in attribution["channels"]}
    assert attribution["chunk_id"] == chunk_id
    assert scores == {"bm25": 8.25, "vector": 0.12, "summary": 0.21}


@pytest.mark.asyncio
async def test_summary_search_failure_keeps_vector_chunks():
    vector = MagicMock()

    async def search(collection_name, *_args, **_kwargs):
        if collection_name == "DocumentChunk_text":
            return [_result("non-uuid", "Vector passage", 0.1)]
        if collection_name == "TextSummary_text":
            raise RuntimeError("summary search unavailable")
        return []

    vector.search = AsyncMock(side_effect=search)

    result = await _retrieve(vector, _bm25([]))

    assert result["chunks"][0].payload["text"] == "Vector passage"
    assert result["chunk_summaries"] == {}


@pytest.mark.asyncio
async def test_summary_load_failure_keeps_vector_chunks():
    chunk_id = str(uuid4())
    vector = MagicMock()

    async def search(collection_name, *_args, **_kwargs):
        if collection_name == "DocumentChunk_text":
            return [_result(chunk_id, "Vector passage", 0.1)]
        return []

    vector.search = AsyncMock(side_effect=search)
    vector.retrieve = AsyncMock(side_effect=RuntimeError("summary load unavailable"))

    result = await _retrieve(vector, _bm25([]))

    assert result["chunks"][0].payload["text"] == "Vector passage"
    assert result["chunk_summaries"] == {}


@pytest.mark.asyncio
async def test_summary_source_chunk_load_failure_keeps_existing_vector_chunks():
    existing_chunk_id = str(uuid4())
    missing_chunk_id = str(uuid4())
    vector = MagicMock()

    async def search(collection_name, *_args, **_kwargs):
        if collection_name == "DocumentChunk_text":
            return [_result(existing_chunk_id, "Existing passage", 0.1)]
        if collection_name == "TextSummary_text":
            return [
                _result(
                    str(uuid4()),
                    "Orphan summary",
                    0.2,
                    source_chunk_id=missing_chunk_id,
                )
            ]
        return []

    vector.search = AsyncMock(side_effect=search)
    vector.retrieve = AsyncMock(side_effect=RuntimeError("source chunk load unavailable"))

    result = await _retrieve(vector, _bm25([]))

    assert result["chunks"][0].payload["text"] == "Existing passage"


@pytest.mark.asyncio
async def test_deleted_cached_bm25_candidate_is_dropped():
    chunk_id = str(uuid4())
    vector = MagicMock()
    vector.retrieve = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.hybrid.chunks.BM25ChunksRetriever",
        return_value=_bm25([({"id": chunk_id, "text": "Deleted passage"}, 4.0)]),
    ):
        results = await search_bm25_chunks(vector, "deleted", 2, None, "OR")

    assert results == []


@pytest.mark.asyncio
async def test_cached_bm25_candidate_uses_current_payload_and_preserves_score():
    chunk_id = str(uuid4())
    vector = MagicMock()
    vector.retrieve = AsyncMock(
        return_value=[_result(chunk_id, "Updated passage", 0.0, revision="current")]
    )

    with patch(
        "cognee.modules.retrieval.hybrid.chunks.BM25ChunksRetriever",
        return_value=_bm25([({"id": chunk_id, "text": "Stale passage", "revision": "old"}, 6.5)]),
    ):
        results = await search_bm25_chunks(vector, "passage", 2, None, "OR")

    assert results == [({"id": chunk_id, "text": "Updated passage", "revision": "current"}, 6.5)]


@pytest.mark.asyncio
async def test_bm25_validation_failure_keeps_vector_lane_results():
    chunk_id = str(uuid4())
    vector = MagicMock()

    async def search(collection_name, *_args, **_kwargs):
        if collection_name == "DocumentChunk_text":
            return [_result(chunk_id, "Vector passage", 0.1)]
        return []

    vector.search = AsyncMock(side_effect=search)
    vector.retrieve = AsyncMock(side_effect=RuntimeError("validation unavailable"))
    bm25 = _bm25([({"id": chunk_id, "text": "Cached passage"}, 5.0)])

    result = await _retrieve(vector, bm25)

    assert result["chunks"][0].payload["text"] == "Vector passage"
    assert [channel["channel"] for channel in result["chunk_attribution"][0]["channels"]] == [
        "vector"
    ]


@pytest.mark.asyncio
async def test_bm25_validation_rechecks_nodeset_scope_on_current_payload():
    chunk_id = str(uuid4())
    vector = MagicMock()
    vector.retrieve = AsyncMock(
        return_value=[_result(chunk_id, "Moved passage", 0.0, belongs_to_set=["scope-b"])]
    )

    with patch(
        "cognee.modules.retrieval.hybrid.chunks.BM25ChunksRetriever",
        return_value=_bm25(
            [
                (
                    {
                        "id": chunk_id,
                        "text": "Previously scoped passage",
                        "belongs_to_set": ["scope-a"],
                    },
                    5.0,
                )
            ]
        ),
    ):
        results = await search_bm25_chunks(
            vector,
            "passage",
            2,
            ["scope-a"],
            "OR",
        )

    assert results == []
