"""DOCUMENTS_LEXICAL: BM25 over the cached documents, best passage as text (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.bm25_documents_retriever import (
    BM25DocumentsRetriever,
    best_passage,
)

lexical_module = sys.modules["cognee.modules.retrieval.lexical_retriever"]

LONG_TEXT = (
    "Alice followed the white rabbit down the hole.\n\n"
    "The Queen of Hearts ruled the garden with a loud voice.\n\n"
    "Tea with the Hatter never ended, the clocks had stopped."
)


def test_best_passage_picks_the_densest_window_and_falls_back_to_the_first():
    assert best_passage(LONG_TEXT, ["queen", "garden"]) == (
        "The Queen of Hearts ruled the garden with a loud voice."
    )
    assert best_passage(LONG_TEXT, []) == "Alice followed the white rabbit down the hole."
    assert best_passage(LONG_TEXT, ["nothing"]) == "Alice followed the white rabbit down the hole."
    assert best_passage("", ["x"]) == ""


@pytest.mark.asyncio
async def test_the_native_index_ranks_and_the_hit_carries_a_snippet():
    doc_id = uuid4()
    engine = SimpleNamespace(
        supports_lexical_search=True,
        lexical_search=AsyncMock(
            return_value=[
                ScoredResult(
                    id=doc_id,
                    score=4.2,
                    payload={
                        "name": "alice.txt",
                        "text": LONG_TEXT,
                        "document_id": "data-1",
                        "mime_type": "text/plain",
                        "char_count": len(LONG_TEXT),
                        "truncated": False,
                    },
                )
            ]
        ),
    )
    retriever = BM25DocumentsRetriever(top_k=3, with_scores=True)
    with patch.object(lexical_module, "get_vector_engine_async", AsyncMock(return_value=engine)):
        hits = await retriever.get_retrieved_objects("queen garden")

    engine.lexical_search.assert_awaited_once_with("CachedDocument_text", "queen garden", limit=3)
    ((payload, score),) = hits
    assert score == 4.2
    assert payload["id"] == str(doc_id)
    assert payload["name"] == "alice.txt" and payload["document_id"] == "data-1"
    assert payload["text"] == "The Queen of Hearts ruled the garden with a loud voice."
    assert payload["passage_count"] == 3
    assert "truncated" in payload and payload["char_count"] == len(LONG_TEXT)

    completion = await retriever.get_completion_from_context("queen garden", hits, "")
    assert completion[0][0]["name"] == "alice.txt"


@pytest.mark.asyncio
async def test_without_a_backend_index_the_cached_nodes_are_scored_in_memory():
    nodes = [
        (
            "cached-1",
            {"id": "cached-1", "type": "CachedDocument", "name": "alice.txt", "text": LONG_TEXT},
        ),
        (
            "cached-2",
            {
                "id": "cached-2",
                "type": "CachedDocument",
                "name": "other.txt",
                "text": "A memo about budgets and spreadsheets.\n\nNothing about rabbits.",
            },
        ),
    ]
    graph_engine = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(nodes, [])))
    vector_engine = SimpleNamespace(supports_lexical_search=False)
    retriever = BM25DocumentsRetriever(top_k=2)
    with (
        patch.object(
            lexical_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
        ),
        patch.object(lexical_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
    ):
        hits = await retriever.get_retrieved_objects("queen garden")

    graph_engine.get_filtered_graph_data.assert_awaited_once_with([{"type": ["CachedDocument"]}])
    assert hits[0]["name"] == "alice.txt"
    assert hits[0]["text"] == "The Queen of Hearts ruled the garden with a loud voice."
