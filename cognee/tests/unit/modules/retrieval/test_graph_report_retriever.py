"""Unit tests for GraphReportRetriever.

Fixtures mirror the real shape returned by ``get_graph_data()``:
nodes are ``(id, props)`` with a ``type`` field; node_set membership is carried
by ``belongs_to_set`` edges pointing at ``NodeSet`` container nodes; edges are
``(src, tgt, relationship_name, props)``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest


# ---------------------------------------------------------------------------
# Fixtures — real get_graph_data() shape
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_graph_data():
    """Two node_sets, a bridge entity (``widget`` in both), and structural edges.

    Exercises hub ranking, cross-set detection, and edge provenance.
    """
    nodes = [
        ("ns_biz", {"name": "business", "type": "NodeSet"}),
        ("ns_eng", {"name": "engineering", "type": "NodeSet"}),
        ("acme", {"name": "Acme", "type": "Entity"}),
        ("widget", {"name": "Widget", "type": "Entity"}),
        ("python", {"name": "Python", "type": "Entity"}),
        ("company", {"name": "Company", "type": "EntityType"}),
        ("chunk1", {"type": "DocumentChunk"}),  # no name -> fallback label
    ]
    edges = [
        # Membership scaffolding (belongs_to_set -> NodeSet container).
        ("acme", "ns_biz", "belongs_to_set", {}),
        ("widget", "ns_biz", "belongs_to_set", {}),
        ("widget", "ns_eng", "belongs_to_set", {}),  # widget bridges both sets
        ("python", "ns_eng", "belongs_to_set", {}),
        ("chunk1", "ns_biz", "belongs_to_set", {}),
        # Entity-to-entity knowledge (EXTRACTED) — these cross node sets.
        ("acme", "widget", "produces", {}),
        ("widget", "python", "built_with", {}),
        # Structural scaffolding (DERIVED).
        ("chunk1", "acme", "contains", {}),
        ("acme", "company", "is_a", {}),
    ]
    return nodes, edges


@pytest.fixture()
def empty_graph_data():
    return [], []


# ---------------------------------------------------------------------------
# get_retrieved_objects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_retrieved_objects_delegates_to_graph_engine(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    mock_engine = AsyncMock()
    mock_engine.get_graph_data.return_value = sample_graph_data

    with patch(
        "cognee.modules.retrieval.graph_report_retriever.get_graph_engine",
        return_value=mock_engine,
    ):
        retriever = GraphReportRetriever(top_n=3)
        result = await retriever.get_retrieved_objects(query="irrelevant")

    assert result == sample_graph_data
    mock_engine.get_graph_data.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_context_from_objects — structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_contains_all_three_sections(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    ctx = await GraphReportRetriever(top_n=5).get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )

    assert "Hub Nodes" in ctx
    assert "Surprising Cross-Set Connections" in ctx
    assert "Edge Provenance" in ctx


@pytest.mark.asyncio
async def test_empty_graph_returns_empty_message(empty_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    ctx = await GraphReportRetriever(top_n=5).get_context_from_objects(
        query="", retrieved_objects=empty_graph_data
    )

    assert "empty" in ctx.lower()


@pytest.mark.asyncio
async def test_hub_nodes_exclude_containers_and_resolve_names(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    ctx = await GraphReportRetriever(top_n=5).get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )
    hub_section = ctx.split("Hub Nodes")[1].split("Surprising")[0]
    # Real entities are surfaced by name...
    assert "**Acme**" in hub_section and "**Widget**" in hub_section
    # ...NodeSet containers are never listed as "god nodes" (they may still
    # appear as a `set:` label, but never as a ranked hub entry).
    assert "**business**" not in hub_section and "**engineering**" not in hub_section


@pytest.mark.asyncio
async def test_cross_set_connections_are_detected(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    ctx = await GraphReportRetriever(top_n=10).get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )
    xset = ctx.split("Surprising Cross-Set Connections")[1].split("Edge Provenance")[0]
    # widget bridges business+engineering, so its links are cross-set.
    assert "Widget" in xset
    assert "business" in xset and "engineering" in xset
    assert "No cross-node_set connections" not in xset


@pytest.mark.asyncio
async def test_edge_provenance_extracted_vs_derived(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    ctx = await GraphReportRetriever(top_n=5).get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )
    prov = ctx.split("Edge Provenance")[1]
    assert "EXTRACTED" in prov and "DERIVED" in prov
    # Regression guard: relationship names must never masquerade as provenance.
    assert "CONTAINS" not in prov and "UNKNOWN" not in prov


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def test_resolve_node_sets_reads_belongs_to_set_edges(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import _resolve_node_sets

    nodes, edges = sample_graph_data
    node_type = {nid: p.get("type") for nid, p in nodes}
    node_sets = _resolve_node_sets(nodes, edges, node_type)

    assert node_sets["acme"] == {"business"}
    assert node_sets["widget"] == {"business", "engineering"}  # bridge entity
    assert node_sets["python"] == {"engineering"}


def test_edge_provenance_counts(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import _edge_provenance

    nodes, edges = sample_graph_data
    node_type = {nid: p.get("type") for nid, p in nodes}
    graph = nx.DiGraph()
    for src, tgt, rel, _ in edges:
        if rel != "belongs_to_set":
            graph.add_edge(src, tgt, relationship=rel)

    counts = _edge_provenance(graph, node_type)
    # produces + built_with are entity<->entity; contains + is_a are structural.
    assert counts == {"EXTRACTED": 2, "DERIVED": 2}


def test_rank_hubs_falls_back_to_degree_without_pagerank():
    from cognee.modules.retrieval.graph_report_retriever import _rank_hubs

    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    node_type = {"a": "Entity", "b": "Entity", "c": "Entity"}
    degree = dict(graph.degree())

    # Empty pagerank simulates scipy-unavailable fallback.
    hubs = _rank_hubs(graph, node_type, degree, {}, top_n=1)
    assert hubs == ["a"]  # highest degree still wins


# ---------------------------------------------------------------------------
# get_completion_from_context — LLM call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_includes_suggested_questions(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=3)
    ctx = await retriever.get_context_from_objects(query="", retrieved_objects=sample_graph_data)

    mock_resp = MagicMock()
    mock_resp.questions = ["What connects Acme and Python?", "What is Widget's role?"]

    with patch(
        "cognee.modules.retrieval.graph_report_retriever.LLMGateway.acreate_structured_output",
        new=AsyncMock(return_value=mock_resp),
    ):
        completion = await retriever.get_completion_from_context(
            query="test", retrieved_objects=sample_graph_data, context=ctx
        )

    assert isinstance(completion, list) and len(completion) == 1
    assert "Suggested Questions" in completion[0]
    assert "What connects Acme and Python?" in completion[0]


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_default_question(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=3)
    ctx = await retriever.get_context_from_objects(query="", retrieved_objects=sample_graph_data)

    with patch(
        "cognee.modules.retrieval.graph_report_retriever.LLMGateway.acreate_structured_output",
        new=AsyncMock(side_effect=RuntimeError("LLM unavailable")),
    ):
        completion = await retriever.get_completion_from_context(
            query="", retrieved_objects=sample_graph_data, context=ctx
        )

    assert isinstance(completion, list)
    assert "Suggested Questions" in completion[0]


# ---------------------------------------------------------------------------
# SearchType registration
# ---------------------------------------------------------------------------


def test_graph_report_is_in_search_type_enum():
    from cognee.modules.search.types import SearchType

    assert hasattr(SearchType, "GRAPH_REPORT")
    assert SearchType.GRAPH_REPORT.value == "GRAPH_REPORT"


@pytest.mark.asyncio
async def test_registry_maps_graph_report_to_retriever():
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever
    from cognee.modules.search.methods.get_search_type_retriever_instance import (
        get_search_type_retriever_instance,
    )
    from cognee.modules.search.types import SearchType

    mock_engine = AsyncMock()
    mock_engine.get_graph_data.return_value = ([], [])

    with patch(
        "cognee.modules.retrieval.graph_report_retriever.get_graph_engine",
        return_value=mock_engine,
    ):
        retriever = await get_search_type_retriever_instance(SearchType.GRAPH_REPORT, query_text="")

    assert isinstance(retriever, GraphReportRetriever)
