"""Unit tests for GraphReportRetriever."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_graph_data():
    """Minimal (nodes, edges) covering hub detection and cross-set links."""
    nodes = [
        ("node_a", {"name": "Alpha", "node_set_name": "set_one"}),
        ("node_b", {"name": "Beta", "node_set_name": "set_one"}),
        ("node_c", {"name": "Gamma", "node_set_name": "set_two"}),
        ("node_d", {"name": "Delta", "node_set_name": "set_two"}),
    ]
    edges = [
        ("node_a", "node_b", "relates_to", {"confidence": "EXTRACTED"}),
        ("node_a", "node_c", "links_to", {"confidence": "INFERRED"}),   # cross-set
        ("node_b", "node_d", "connects", {"confidence": "EXTRACTED"}),  # cross-set
        ("node_c", "node_d", "follows", {}),
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

    retriever = GraphReportRetriever(top_n=3)
    ctx = await retriever.get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )

    assert "Hub Nodes" in ctx
    assert "Surprising Cross-Set Connections" in ctx
    assert "Confidence Tags" in ctx


@pytest.mark.asyncio
async def test_empty_graph_returns_empty_message(empty_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=5)
    ctx = await retriever.get_context_from_objects(
        query="", retrieved_objects=empty_graph_data
    )

    assert "empty" in ctx.lower()


@pytest.mark.asyncio
async def test_cross_set_connections_are_detected(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=10)
    ctx = await retriever.get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )
    # node_a (set_one) → node_c (set_two) must appear
    assert "set_one" in ctx and "set_two" in ctx


@pytest.mark.asyncio
async def test_confidence_tags_appear_in_context(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=5)
    ctx = await retriever.get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )
    assert "EXTRACTED" in ctx
    assert "INFERRED" in ctx


# ---------------------------------------------------------------------------
# get_completion_from_context — LLM call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_includes_suggested_questions(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=3)
    ctx = await retriever.get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )

    mock_resp = MagicMock()
    mock_resp.questions = ["What connects Alpha and Gamma?", "What is Beta's role?"]

    with patch(
        "cognee.modules.retrieval.graph_report_retriever.LLMGateway.acreate_structured_output",
        new=AsyncMock(return_value=mock_resp),
    ):
        completion = await retriever.get_completion_from_context(
            query="test", retrieved_objects=sample_graph_data, context=ctx
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert "Suggested Questions" in completion[0]
    assert "What connects Alpha and Gamma?" in completion[0]


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_default_question(sample_graph_data):
    from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

    retriever = GraphReportRetriever(top_n=3)
    ctx = await retriever.get_context_from_objects(
        query="", retrieved_objects=sample_graph_data
    )

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

    # Patch graph engine so the retriever can be instantiated without a live DB
    mock_engine = AsyncMock()
    mock_engine.get_graph_data.return_value = ([], [])

    with patch(
        "cognee.modules.retrieval.graph_report_retriever.get_graph_engine",
        return_value=mock_engine,
    ):
        retriever = await get_search_type_retriever_instance(
            SearchType.GRAPH_REPORT, query_text=""
        )

    assert isinstance(retriever, GraphReportRetriever)
