"""Lexical retrievers use the vector store's BM25 index only where it ranks every row (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.jaccard_retrival import JaccardChunksRetriever

lexical_module = sys.modules["cognee.modules.retrieval.lexical_retriever"]


def _vector_engine(*, supports=True, indexed=True, results=None):
    return SimpleNamespace(
        supports_lexical_search=supports,
        has_lexical_index=AsyncMock(return_value=indexed),
        lexical_search=AsyncMock(return_value=results or []),
    )


def _graph_engine(nodes):
    return SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(nodes, [])))


async def _retrieve(retriever, query, vector_engine, graph_engine):
    with (
        patch.object(
            lexical_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
        ),
        patch.object(lexical_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
        patch.object(lexical_module, "attach_chunk_conflicts", AsyncMock()) as conflicts,
    ):
        hits = await retriever.get_retrieved_objects(query)
    return hits, conflicts


@pytest.mark.asyncio
async def test_an_indexed_collection_answers_from_the_store_and_the_graph_is_not_loaded():
    chunk_id = uuid4()
    vector_engine = _vector_engine(
        results=[
            ScoredResult(id=chunk_id, score=2.5, payload={"text": "rabbit hole", "chunk_index": 0})
        ]
    )
    graph_engine = _graph_engine([])

    hits, conflicts = await _retrieve(
        BM25ChunksRetriever(top_k=5, with_scores=True), "rabbit", vector_engine, graph_engine
    )

    vector_engine.has_lexical_index.assert_awaited_once_with("DocumentChunk_text")
    vector_engine.lexical_search.assert_awaited_once_with("DocumentChunk_text", "rabbit", limit=5)
    graph_engine.get_filtered_graph_data.assert_not_awaited()
    assert hits == [({"text": "rabbit hole", "chunk_index": 0, "id": str(chunk_id)}, 2.5)]
    conflicts.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_collection_without_an_index_is_scored_in_memory():
    """Chunks stored before the collection had a lexical index (or a type that
    declares none) must not lose rows to a partial index: memory scoring answers."""
    nodes = [("c1", {"type": "DocumentChunk", "text": "the rabbit hole"})]
    vector_engine = _vector_engine(indexed=False)

    hits, _ = await _retrieve(
        BM25ChunksRetriever(top_k=1), "rabbit", vector_engine, _graph_engine(nodes)
    )

    vector_engine.has_lexical_index.assert_awaited_once_with("DocumentChunk_text")
    vector_engine.lexical_search.assert_not_awaited()
    assert [hit["id"] for hit in hits] == ["c1"]


@pytest.mark.asyncio
async def test_a_backend_without_lexical_support_is_scored_in_memory():
    nodes = [
        ("c1", {"type": "DocumentChunk", "text": "the rabbit hole"}),
        ("c2", {"type": "DocumentChunk", "text": "a tea party"}),
    ]
    vector_engine = _vector_engine(supports=False)

    hits, _ = await _retrieve(
        BM25ChunksRetriever(top_k=1), "rabbit", vector_engine, _graph_engine(nodes)
    )

    vector_engine.has_lexical_index.assert_not_awaited()
    vector_engine.lexical_search.assert_not_awaited()
    assert [hit["id"] for hit in hits] == ["c1"]


@pytest.mark.asyncio
async def test_the_jaccard_scorer_never_uses_the_index():
    vector_engine = _vector_engine()
    nodes = [("c1", {"type": "DocumentChunk", "text": "rabbit"})]

    hits, _ = await _retrieve(
        JaccardChunksRetriever(top_k=1), "rabbit", vector_engine, _graph_engine(nodes)
    )

    vector_engine.lexical_search.assert_not_awaited()
    assert [hit["id"] for hit in hits] == ["c1"]
