"""CHUNKS_LEXICAL runs in the vector store's lexical index when the backend has one (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.jaccard_retrival import JaccardChunksRetriever

lexical_module = sys.modules["cognee.modules.retrieval.lexical_retriever"]


@pytest.mark.asyncio
async def test_chunks_come_from_the_index_and_the_graph_is_not_loaded():
    chunk_id = uuid4()
    vector_engine = SimpleNamespace(
        supports_lexical_search=True,
        lexical_search=AsyncMock(
            return_value=[
                ScoredResult(
                    id=chunk_id, score=2.5, payload={"text": "rabbit hole", "chunk_index": 0}
                )
            ]
        ),
    )
    graph_engine = SimpleNamespace(get_filtered_graph_data=AsyncMock())
    retriever = BM25ChunksRetriever(top_k=5, with_scores=True)
    with (
        patch.object(
            lexical_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
        ),
        patch.object(lexical_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
        patch.object(lexical_module, "attach_chunk_conflicts", AsyncMock()) as conflicts,
    ):
        hits = await retriever.get_retrieved_objects("rabbit")

    vector_engine.lexical_search.assert_awaited_once_with("DocumentChunk_text", "rabbit", limit=5)
    graph_engine.get_filtered_graph_data.assert_not_awaited()
    assert hits == [({"text": "rabbit hole", "chunk_index": 0, "id": str(chunk_id)}, 2.5)]
    conflicts.assert_awaited_once()


@pytest.mark.asyncio
async def test_without_the_index_the_chunk_nodes_are_scored_in_memory():
    nodes = [
        ("c1", {"type": "DocumentChunk", "text": "the rabbit hole"}),
        ("c2", {"type": "DocumentChunk", "text": "a tea party"}),
    ]
    vector_engine = SimpleNamespace(supports_lexical_search=False, lexical_search=AsyncMock())
    graph_engine = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(nodes, [])))
    retriever = BM25ChunksRetriever(top_k=1)
    with (
        patch.object(
            lexical_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
        ),
        patch.object(lexical_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
        patch.object(lexical_module, "attach_chunk_conflicts", AsyncMock()),
    ):
        hits = await retriever.get_retrieved_objects("rabbit")

    vector_engine.lexical_search.assert_not_awaited()
    assert [hit["id"] for hit in hits] == ["c1"]


@pytest.mark.asyncio
async def test_the_jaccard_scorer_never_uses_the_index():
    vector_engine = SimpleNamespace(supports_lexical_search=True, lexical_search=AsyncMock())
    nodes = [("c1", {"type": "DocumentChunk", "text": "rabbit"})]
    graph_engine = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(nodes, [])))
    retriever = JaccardChunksRetriever(top_k=1)
    with (
        patch.object(
            lexical_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
        ),
        patch.object(lexical_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
        patch.object(lexical_module, "attach_chunk_conflicts", AsyncMock()),
    ):
        hits = await retriever.get_retrieved_objects("rabbit")

    vector_engine.lexical_search.assert_not_awaited()
    assert [hit["id"] for hit in hits] == ["c1"]
