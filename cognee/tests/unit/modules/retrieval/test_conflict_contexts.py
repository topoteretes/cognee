"""Stored conflict context stays attached to the selected hits and preserves result shapes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.fact_conflicts import CONFLICT_CITES
from cognee.modules.retrieval import chunks_retriever, completion_retriever, summaries_retriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.lexical_retriever import LexicalRetriever, tokenize_words
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.modules.retrieval.utils import conflict_context
from cognee.modules.retrieval.utils.conflict_context import get_chunk_conflicts

EXPLANATION = "Bob replaced Alice as Acme's CEO."


@pytest.fixture
def graph():
    return SimpleNamespace(
        get_neighborhood=AsyncMock(
            return_value=(
                [("f1", {"type": "FactConflict", "text": EXPLANATION})],
                [
                    (
                        "f1",
                        chunk_id,
                        "conflict_cites",
                        {"document": "board.txt", "effective_date": "2026-01-10"},
                    )
                    for chunk_id in ("c1", "c2")
                ],
            )
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["rag", "chunks", "summaries"])
@pytest.mark.parametrize("graph_available", [True, False])
async def test_chunk_contexts_and_raw_results(kind, graph_available, graph, monkeypatch):
    modules = {
        "rag": completion_retriever,
        "chunks": chunks_retriever,
        "summaries": summaries_retriever,
    }
    classes = {
        "rag": completion_retriever.CompletionRetriever,
        "chunks": chunks_retriever.ChunksRetriever,
        "summaries": summaries_retriever.SummariesRetriever,
    }
    rows = [
        SimpleNamespace(
            id=chunk_id,
            payload={"id": chunk_id, "text": f"Passage {chunk_id}", "source": "original source"},
            score=0.2,
        )
        for chunk_id in ("c1", "c2")
    ]
    if kind == "summaries":
        for row in rows:
            row.payload["source_chunk_id"] = row.id
            row.payload["id"] = f"summary-{row.id}"
    if not graph_available:
        graph.get_neighborhood.side_effect = RuntimeError("Graph unavailable")
    vector = SimpleNamespace(search=AsyncMock(return_value=rows))
    module = modules[kind]
    if kind == "rag":
        monkeypatch.setattr(module, "get_vector_engine_async", AsyncMock(return_value=vector))
        monkeypatch.setattr(conflict_context, "get_graph_engine", AsyncMock(return_value=graph))
        monkeypatch.setattr(module, "load_preference_weights", AsyncMock(return_value={}))
    else:
        monkeypatch.setattr(
            module,
            "get_unified_engine",
            AsyncMock(return_value=SimpleNamespace(graph=graph, vector=vector)),
        )
    retriever = classes[kind](top_k=2)
    objects = await retriever.get_retrieved_objects("CEO")
    context = await retriever.get_context_from_objects("CEO", objects)
    assert objects is rows
    assert all(row.payload["source"] == "original source" for row in objects)
    if graph_available:
        assert context.count(EXPLANATION) == 1
        assert context.count("source: board.txt (2026-01-10)") == 2
        assert "## Fact conflicts" in context
    else:
        assert context == "Passage c1\nPassage c2"
    if kind != "rag":
        result = await retriever.get_completion_from_context("CEO", objects, context)
        assert [row["score"] for row in result] == [0.2, 0.2]
        assert all("_passage_header" not in row for row in result)
        assert all(row["source"] == "original source" for row in result)
        assert all(
            row.get("conflicts", []) == ([EXPLANATION] if graph_available else []) for row in result
        )
        if graph_available:
            assert rows[0].payload["_passage_header"] == "board.txt (2026-01-10)"


@pytest.mark.asyncio
async def test_completion_merge_keeps_each_hits_passage_header():
    retriever = completion_retriever.CompletionRetriever(top_k=2)
    primary, secondary = [
        [SimpleNamespace(id=chunk_id, payload={"text": text, "_passage_header": source})]
        for chunk_id, text, source in (("c1", "First", "Report A"), ("c2", "Second", "Report B"))
    ]
    merged = retriever.merge_retrieved_objects(primary, secondary)
    assert await retriever.get_context_from_objects("CEO", merged) == (
        "source: Report A\nFirst\nsource: Report B\nSecond"
    )


@pytest.mark.asyncio
async def test_summary_without_source_chunk_id_keeps_existing_shape(graph, monkeypatch):
    row = SimpleNamespace(id="c1", payload={"id": "c1", "text": "Summary"}, score=0.1)
    engine = SimpleNamespace(
        graph=graph, vector=SimpleNamespace(search=AsyncMock(return_value=[row]))
    )
    monkeypatch.setattr(summaries_retriever, "get_unified_engine", AsyncMock(return_value=engine))
    objects = await summaries_retriever.SummariesRetriever().get_retrieved_objects("CEO")
    assert objects[0].payload == {"id": "c1", "text": "Summary"}
    graph.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_scores", [True, False])
async def test_lexical_annotations_do_not_modify_cached_payloads(with_scores, graph, monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.retrieval.utils.conflict_context.get_graph_engine",
        AsyncMock(return_value=graph),
    )
    retriever = LexicalRetriever(tokenize_words, lambda query, chunk: 1.0, with_scores=with_scores)
    original_payload = {"id": "c1", "text": "Acme CEO", "source": "original source"}
    retriever.payloads = {"c1": original_payload.copy()}
    retriever.chunks = {"c1": ["acme", "ceo"]}
    retriever._initialized = True
    objects = await retriever.get_retrieved_objects("CEO")
    context = await retriever.get_context_from_objects("CEO", objects)
    assert "source: board.txt (2026-01-10)\nAcme CEO" in context
    assert context.count(EXPLANATION) == 1
    result = await retriever.get_completion_from_context("CEO", objects, context)
    payload = result[0][0] if with_scores else result[0]
    assert payload["conflicts"] == [EXPLANATION]
    assert "_passage_header" not in payload
    assert payload["source"] == "original source"
    if with_scores:
        assert result[0][1] == 1.0
    assert retriever.payloads == {"c1": original_payload}
    graph.get_neighborhood.return_value = ([], [])
    fresh = await retriever.get_retrieved_objects("CEO")
    fresh_payload = fresh[0][0] if with_scores else fresh[0]
    assert "conflicts" not in fresh_payload and "_passage_header" not in fresh_payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retriever_type",
    [
        GraphCompletionRetriever,
        GraphCompletionCotRetriever,
        GraphCompletionContextExtensionRetriever,
        TemporalRetriever,
    ],
)
async def test_graph_family_uses_chunk_conflict_block_with_lazy_engine(
    retriever_type, graph, monkeypatch
):
    get_engine = AsyncMock(return_value=SimpleNamespace(graph=graph))
    monkeypatch.setattr(
        "cognee.modules.retrieval.graph_completion_retriever.get_unified_engine", get_engine
    )
    edge = Edge(
        Node("c1", {"type": "DocumentChunk", "text": "Source text"}),
        Node("acme", {"type": "Entity", "name": "Acme"}),
        attributes={
            "relationship_name": "contains",
            "edge_text": "Bob leads Acme",
            "conflict_marks": [{"conflict_id": "f1", "status": "current"}],
        },
    )
    context = await retriever_type().resolve_edges_to_text([edge])
    assert "[current]" in context
    assert context.endswith(f"## Fact conflicts\n- {EXPLANATION}")
    get_engine.assert_awaited_once()
    graph.get_neighborhood.assert_awaited_once_with(["c1"], depth=1, edge_types=["conflict_cites"])


@pytest.mark.asyncio
async def test_graph_context_survives_engine_failure(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.retrieval.graph_completion_retriever.get_unified_engine",
        AsyncMock(side_effect=RuntimeError("unavailable")),
    )
    edge = Edge(
        Node("c1", {"type": "DocumentChunk", "text": "Source"}),
        Node("acme", {"name": "Acme"}),
        attributes={"relationship_name": "contains"},
    )
    context = await GraphCompletionRetriever().resolve_edges_to_text([edge])
    assert "Connections:" in context and "## Fact conflicts" not in context


@pytest.mark.asyncio
async def test_chunk_conflicts_filter_induced_edges_and_keep_all_reached_datasets():
    graph = AsyncMock()
    graph.get_neighborhood.return_value = (
        [
            ("f1", {"text": "First conflict", "dataset_id": "one"}),
            ("f2", {"text": "Second conflict", "dataset_id": "two"}),
        ],
        [
            ("f2", "c1", CONFLICT_CITES, {"document": "report", "effective_date": "2026-01-10"}),
            ("f1", "c1", CONFLICT_CITES, {"document": "report", "effective_date": "2026-01-10"}),
            ("f1", "c2", "conflict_about", {"edge_text": "not a citation"}),
            ("f1", "unrequested", CONFLICT_CITES, {"document": "outside"}),
        ],
    )
    result = await get_chunk_conflicts(graph, ["c1", "c2"])
    assert set(result) == {"c1"}
    assert result["c1"].document == "report"
    assert result["c1"].effective_date == "2026-01-10"
    assert result["c1"].conflicts == {"f1": "First conflict", "f2": "Second conflict"}
    graph.get_neighborhood.assert_awaited_once_with(
        ["c1", "c2"], depth=1, edge_types=[CONFLICT_CITES]
    )


@pytest.mark.asyncio
async def test_empty_or_unavailable_graph_leaves_chunks_unchanged():
    graph = AsyncMock()
    assert await get_chunk_conflicts(graph, []) == {}
    graph.get_neighborhood.assert_not_awaited()
    graph.get_neighborhood.side_effect = RuntimeError("offline")
    assert await get_chunk_conflicts(graph, ["c1"]) == {}
