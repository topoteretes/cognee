"""DOCUMENTS_LEXICAL: BM25 over the cached documents, best passage as text (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.engine.models.CachedDocument import CachedDocument
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


def _cached(name, text, node_sets=()):
    return CachedDocument(
        name=name, text=text, document_id=str(uuid4()), belongs_to_set=list(node_sets)
    )


def _lancedb_row(document: CachedDocument, score: float) -> ScoredResult:
    """A hit as LanceDB returns it: the index schema's payload, not the whole point."""
    payload = LanceDBAdapter._to_index_point(document, document.text).model_dump()
    return ScoredResult(id=document.id, score=score, payload=payload)


def _pgvector_row(document: CachedDocument, score: float) -> ScoredResult:
    """A hit as pgvector returns it: the whole point's payload."""
    return ScoredResult(id=document.id, score=score, payload=document.model_dump(mode="json"))


def _native_engine(rows):
    return SimpleNamespace(
        supports_lexical_search=True,
        has_lexical_index=AsyncMock(return_value=True),
        lexical_search=AsyncMock(return_value=rows),
    )


def test_best_passage_picks_the_densest_window_and_falls_back_to_the_first():
    assert best_passage(LONG_TEXT, ["queen", "garden"]) == (
        "The Queen of Hearts ruled the garden with a loud voice."
    )
    assert best_passage(LONG_TEXT, []) == "Alice followed the white rabbit down the hole."
    assert best_passage(LONG_TEXT, ["nothing"]) == "Alice followed the white rabbit down the hole."
    assert best_passage("", ["x"]) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [_lancedb_row, _pgvector_row], ids=["lancedb", "pgvector"])
async def test_a_native_hit_has_the_same_shape_on_every_backend(row):
    document = _cached("alice.txt", LONG_TEXT, ["project"])
    engine = _native_engine([row(document, 4.2)])
    retriever = BM25DocumentsRetriever(top_k=3, with_scores=True)
    with patch.object(lexical_module, "get_vector_engine_async", AsyncMock(return_value=engine)):
        hits = await retriever.get_retrieved_objects("queen garden")

    engine.lexical_search.assert_awaited_once_with("CachedDocument_text", "queen garden", limit=3)
    ((payload, score),) = hits
    assert score == 4.2
    assert payload == {
        "id": str(document.id),
        "document_id": document.document_id,
        "belongs_to_set": ["project"],
        "name": "alice.txt",
        "text": "The Queen of Hearts ruled the garden with a loud voice.",
        "passage_count": 3,
    }

    completion = await retriever.get_completion_from_context("queen garden", hits, "")
    assert completion[0][0]["name"] == "alice.txt"


@pytest.mark.asyncio
async def test_a_node_set_filter_keeps_only_that_sets_documents():
    project = _cached("roadmap.txt", "The salary review is in the roadmap.", ["project"])
    personal = _cached("notes.txt", "My personal salary negotiation notes.", ["user"])
    engine = _native_engine([_lancedb_row(personal, 3.0), _lancedb_row(project, 2.0)])
    retriever = BM25DocumentsRetriever(top_k=5, node_name=["project"])
    with patch.object(lexical_module, "get_vector_engine_async", AsyncMock(return_value=engine)):
        hits = await retriever.get_retrieved_objects("salary review")

    # The index ranks the whole collection, so it is asked for more than top_k.
    engine.lexical_search.assert_awaited_once_with("CachedDocument_text", "salary review", limit=50)
    assert [hit["name"] for hit in hits] == ["roadmap.txt"]


@pytest.mark.asyncio
async def test_the_and_operator_needs_every_named_set():
    both = _cached("both.txt", "salary review", ["project", "hr"])
    one = _cached("one.txt", "salary review", ["project"])
    engine = _native_engine([_lancedb_row(one, 3.0), _lancedb_row(both, 2.0)])
    retriever = BM25DocumentsRetriever(
        top_k=5, node_name=["project", "hr"], node_name_filter_operator="AND"
    )
    with patch.object(lexical_module, "get_vector_engine_async", AsyncMock(return_value=engine)):
        hits = await retriever.get_retrieved_objects("salary review")

    assert [hit["name"] for hit in hits] == ["both.txt"]


def _memory_engines(nodes):
    graph_engine = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(nodes, [])))
    vector_engine = SimpleNamespace(supports_lexical_search=False)
    return (
        patch.object(
            lexical_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
        ),
        patch.object(lexical_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
        graph_engine,
    )


def _node(document: CachedDocument):
    return (str(document.id), {**document.model_dump(mode="json"), "id": str(document.id)})


@pytest.mark.asyncio
async def test_without_a_backend_index_the_cached_nodes_are_scored_in_memory():
    alice = _cached("alice.txt", LONG_TEXT)
    other = _cached("other.txt", "A memo about budgets and spreadsheets.\n\nNothing about rabbits.")
    vector_patch, graph_patch, graph_engine = _memory_engines([_node(alice), _node(other)])
    retriever = BM25DocumentsRetriever(top_k=2)
    with vector_patch, graph_patch:
        hits = await retriever.get_retrieved_objects("queen garden")

    graph_engine.get_filtered_graph_data.assert_awaited_once_with([{"type": ["CachedDocument"]}])
    # The memo shares no term with the query: no hit, as on the backend indexes.
    assert [hit["name"] for hit in hits] == ["alice.txt"]
    assert hits[0]["text"] == "The Queen of Hearts ruled the garden with a loud voice."


@pytest.mark.asyncio
async def test_in_memory_a_query_with_no_matching_term_returns_nothing():
    vector_patch, graph_patch, _ = _memory_engines([_node(_cached("alice.txt", LONG_TEXT))])
    retriever = BM25DocumentsRetriever(top_k=5)
    with vector_patch, graph_patch:
        assert await retriever.get_retrieved_objects("kubernetes migration") == []


@pytest.mark.asyncio
async def test_in_memory_the_node_set_filter_applies_before_ranking():
    project = _cached("roadmap.txt", "salary review roadmap", ["project"])
    personal = _cached("notes.txt", "salary review salary review", ["user"])
    vector_patch, graph_patch, _ = _memory_engines([_node(project), _node(personal)])
    retriever = BM25DocumentsRetriever(top_k=1, node_name=["project"])
    with vector_patch, graph_patch:
        hits = await retriever.get_retrieved_objects("salary review")

    assert [hit["name"] for hit in hits] == ["roadmap.txt"]


@pytest.mark.asyncio
async def test_in_memory_a_filter_that_matches_no_set_is_an_empty_result():
    vector_patch, graph_patch, _ = _memory_engines([_node(_cached("a.txt", LONG_TEXT, ["user"]))])
    retriever = BM25DocumentsRetriever(top_k=5, node_name=["project"])
    with vector_patch, graph_patch:
        assert await retriever.get_retrieved_objects("queen") == []
