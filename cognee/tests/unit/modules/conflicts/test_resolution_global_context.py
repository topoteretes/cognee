"""User decisions supplement the global index only for the current retrieved set."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


def _objects(retriever_type, node_id):
    if retriever_type is HybridRetriever:
        return {"chunks": [{"id": node_id, "text": "Retrieved passage"}], "entities": []}
    return [Edge(Node(node_id, {"name": "Retrieved node"}), Node("neighbor"))]


def _retriever(monkeypatch, retriever_type, resolutions, top_k):
    retriever = retriever_type(include_global_context_index=True, global_context_index_top_k=top_k)
    retriever._unified_engine = SimpleNamespace(
        graph=SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(resolutions, []))),
        vector=object(),
    )
    module = importlib.import_module(retriever_type.__module__)
    monkeypatch.setattr(module, "load_root_text", AsyncMock(return_value="Root summary"))
    search_summaries = AsyncMock(return_value=["Area summary"])
    monkeypatch.setattr(module, "search_top_global_context_summaries", search_summaries)
    return retriever, search_summaries


@pytest.mark.asyncio
@pytest.mark.parametrize("retriever_type", [HybridRetriever, GraphCompletionRetriever])
@pytest.mark.parametrize("link_field", ["about_id", "values", "sources"])
async def test_global_context_keeps_summaries_and_states_every_relevant_decision(
    monkeypatch, retriever_type, link_field
):
    def decision(fact, node_id):
        return {
            "resolution_source": "user",
            "selected_fact": fact,
            link_field: node_id if link_field == "about_id" else [node_id],
        }

    resolutions = [
        ("unrelated", decision("Unrelated decision", "elsewhere")),
        ("a", decision("Accepted fact A", "retrieved")),
        ("b", decision("Accepted fact B", "retrieved")),
        ("c", decision("Accepted fact C", "retrieved")),
    ]
    retriever, search_summaries = _retriever(monkeypatch, retriever_type, resolutions, top_k=2)
    context = await retriever.get_context_from_objects(
        query="q", retrieved_objects=_objects(retriever_type, "retrieved")
    )
    assert "Root summary" in context and "Area summary" in context
    if retriever_type is HybridRetriever:
        assert context.startswith("## Global context\n")
    # top_k sizes the summaries only: no decision about the retrieved set is cut.
    for fact in ("A", "B", "C"):
        assert f"Accepted by user: Accepted fact {fact}" in context
    assert "Unrelated decision" not in context
    assert context.count("Accepted by user:") == 3
    search_summaries.assert_awaited_once_with("q", 2, retriever._unified_engine.vector)


@pytest.mark.asyncio
@pytest.mark.parametrize("retriever_type", [HybridRetriever, GraphCompletionRetriever])
async def test_a_decision_whose_selected_fact_is_in_context_is_not_restated(
    monkeypatch, retriever_type
):
    """The selected edge, or the link standing in for it, already states the decision."""

    def decision(fact, fact_id):
        return {
            "resolution_source": "user",
            "selected_fact": fact,
            "selected_fact_id": fact_id,
            "about_id": "retrieved",
        }

    resolutions = [
        ("a", decision("Accepted fact A", "a-fact")),
        ("b", decision("Accepted fact B", "b-fact")),
    ]
    retriever, _ = _retriever(monkeypatch, retriever_type, resolutions, top_k=2)
    edge = {"edge_object_id": "a-fact", "edge_text": "Accepted fact A"}
    if retriever_type is HybridRetriever:
        objects = {
            "chunks": [],
            "entities": [
                {
                    "id": "retrieved",
                    "name": "Retrieved",
                    "edges": [{**edge, "text": edge["edge_text"], "source_id": "retrieved"}],
                }
            ],
        }
    else:
        objects = [Edge(Node("retrieved", {"name": "Retrieved"}), Node("a"), attributes=edge)]
    context = await retriever.get_context_from_objects(query="q", retrieved_objects=objects)
    assert "Accepted by user: Accepted fact B" in context
    assert "Accepted by user: Accepted fact A" not in context


@pytest.mark.asyncio
@pytest.mark.parametrize("retriever_type", [HybridRetriever, GraphCompletionRetriever])
async def test_unrelated_decision_does_not_replace_global_index(monkeypatch, retriever_type):
    retriever, _ = _retriever(
        monkeypatch,
        retriever_type,
        [
            (
                "conflict",
                {
                    "resolution_source": "user",
                    "selected_fact": "CEO is Alice",
                    "about_id": "company",
                },
            )
        ],
        top_k=3,
    )
    context = await retriever.get_context_from_objects(
        query="q", retrieved_objects=_objects(retriever_type, "unrelated")
    )
    assert "Root summary" in context and "Area summary" in context
    assert "CEO is Alice" not in context
    assert "## Accepted facts" not in context


@pytest.mark.asyncio
async def test_batch_decisions_are_selected_for_each_queries_retrieved_objects(monkeypatch):
    retriever, search_summaries = _retriever(
        monkeypatch,
        HybridRetriever,
        [
            (
                "first",
                {
                    "resolution_source": "user",
                    "selected_fact": "First decision",
                    "about_id": "first",
                },
            ),
            (
                "second",
                {
                    "resolution_source": "user",
                    "selected_fact": "Second decision",
                    "sources": ["second"],
                },
            ),
        ],
        top_k=1,
    )
    contexts = await retriever.get_context_from_objects(
        query_batch=["first query", "second query"],
        retrieved_objects=[_objects(HybridRetriever, "first"), _objects(HybridRetriever, "second")],
    )
    for context in contexts:
        assert context.startswith("## Global context\n")
        assert "Root summary" in context and "Area summary" in context
        assert context.count("Accepted by user:") == 1
    assert "First decision" in contexts[0] and "Second decision" not in contexts[0]
    assert "Second decision" in contexts[1] and "First decision" not in contexts[1]
    assert search_summaries.await_count == 2
