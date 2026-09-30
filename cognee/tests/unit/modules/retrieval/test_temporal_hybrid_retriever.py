"""TemporalHybridRetriever and its matching helpers (SDK-828).

Ported from cognee/tests/unit/examples/test_temporal_hybrid_demo.py, where they
exercised the POC copies in examples/advanced_guides/temporal_awareness_example.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.temporal_hybrid.matching import (
    build_temporal_index,
    extract_query_interval,
    filter_hybrid,
    match_temporal_neighborhood,
)
from cognee.modules.retrieval.temporal_hybrid_retriever import TemporalHybridRetriever


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _query_interval(start: dict | None = None, end: dict | None = None):
    from cognee.tasks.temporal_graph.models import QueryInterval
    from cognee.tasks.temporal_graph.models import Timestamp as QueryTime

    return QueryInterval(
        starts_at=None if start is None else QueryTime(**start),
        ends_at=None if end is None else QueryTime(**end),
    )


def _utc_fields(year: int, month: int = 1, day: int = 1) -> dict:
    return {
        "year": year,
        "month": month,
        "day": day,
        "hour": 0,
        "minute": 0,
        "second": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "end", "expected_start", "expected_end", "reason"),
    [
        (_utc_fields(1950), _utc_fields(1951), _utc(1950, 1, 1), _utc(1951, 1, 1), None),
        (_utc_fields(2000), _utc_fields(2007), _utc(2000, 1, 1), _utc(2007, 1, 1), None),
        (None, _utc_fields(1950), None, _utc(1950, 1, 1), None),
        (_utc_fields(1951), None, _utc(1951, 1, 1), None, None),
        (None, None, None, None, "no_time_constraint"),
        (_utc_fields(1951), _utc_fields(1950), None, None, "invalid_interval"),
        (_utc_fields(2021, 2, 29), None, None, None, "invalid_interval"),
    ],
)
async def test_extract_query_interval(
    start, end, expected_start, expected_end, reason, monkeypatch
):

    monkeypatch.setattr(
        "cognee.modules.retrieval.temporal_hybrid.matching.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=_query_interval(start, end)),
    )
    got_start, got_end, got_reason = await extract_query_interval("query")
    assert (got_start, got_end, got_reason) == (expected_start, expected_end, reason)


@pytest.mark.asyncio
async def test_extract_query_interval_propagates_llm_errors(monkeypatch):

    monkeypatch.setattr(
        "cognee.modules.retrieval.temporal_hybrid.matching.LLMGateway.acreate_structured_output",
        AsyncMock(side_effect=RuntimeError("provider")),
    )
    with pytest.raises(RuntimeError, match="provider"):
        await extract_query_interval("in 1950")


def _graph_node(node_id: str, type_name: str, **properties):
    return (node_id, {"type": type_name, **properties})


def _graph_edge(source: str, target: str, relationship: str):
    return (source, target, relationship, {})


def test_match_temporal_neighborhood_overlap_table():

    nodes = [
        _graph_node("chunk_period", "DocumentChunk"),
        _graph_node("chunk_year", "DocumentChunk"),
        _graph_node("chunk_contact", "DocumentChunk"),
        _graph_node("chunk_invalid", "DocumentChunk"),
        _graph_node("chunk_other", "DocumentChunk"),
        _graph_node("ts_1947", "Timestamp", timestamp_str="1947"),
        _graph_node("ts_1956", "Timestamp", timestamp_str="1956"),
        _graph_node("ts_1950", "Timestamp", timestamp_str="1950"),
        _graph_node("ts_1951", "Timestamp", timestamp_str="1951"),
        _graph_node("ts_1949", "Timestamp", timestamp_str="1949"),
        _graph_node("ts_1960", "Timestamp", timestamp_str="1960"),
        _graph_node("ts_bad", "Timestamp", timestamp_str="1940s"),
        _graph_node("presidency", "Entity"),
        _graph_node("missing_end", "Entity"),
        _graph_node("two_begins", "Entity"),
        _graph_node("reversed", "Entity"),
        _graph_node("person", "Entity"),
    ]
    edges = [
        _graph_edge("chunk_period", "presidency", "contains"),
        _graph_edge("chunk_period", "ts_1947", "contains"),
        _graph_edge("chunk_period", "ts_1956", "contains"),
        _graph_edge("chunk_period", "person", "contains"),
        _graph_edge("presidency", "ts_1947", "begins_at"),
        _graph_edge("presidency", "ts_1956", "ends_at"),
        _graph_edge("chunk_year", "ts_1950", "contains"),
        _graph_edge("chunk_contact", "ts_1951", "contains"),
        _graph_edge("chunk_contact", "ts_1949", "contains"),
        _graph_edge("chunk_invalid", "missing_end", "contains"),
        _graph_edge("chunk_invalid", "ts_1960", "contains"),
        _graph_edge("missing_end", "ts_1960", "begins_at"),
        _graph_edge("chunk_invalid", "two_begins", "contains"),
        _graph_edge("two_begins", "ts_1947", "begins_at"),
        _graph_edge("two_begins", "ts_1950", "begins_at"),
        _graph_edge("two_begins", "ts_1956", "ends_at"),
        _graph_edge("chunk_invalid", "reversed", "contains"),
        _graph_edge("reversed", "ts_1960", "begins_at"),
        _graph_edge("reversed", "ts_1947", "ends_at"),
        _graph_edge("chunk_invalid", "ts_bad", "contains"),
        _graph_edge("chunk_other", "person", "contains"),
        _graph_edge("person", "ts_1950", "born_at"),
    ]
    start, end = _utc(1950, 1, 1), _utc(1951, 1, 1)
    matched = match_temporal_neighborhood(build_temporal_index(nodes, edges), start, end)

    assert matched["timestamp_ids"] == {"ts_1950"}
    assert matched["period_ids"] == {"presidency"}
    assert matched["eligible_chunk_ids"] == {"chunk_period", "chunk_year"}
    assert "chunk_contact" not in matched["eligible_chunk_ids"]
    assert "chunk_invalid" not in matched["eligible_chunk_ids"]
    assert "chunk_other" not in matched["eligible_chunk_ids"]
    assert matched["chunk_entities"]["chunk_period"] == {"presidency", "person"}
    assert ("presidency", "ts_1947", "begins_at") in matched["temporal_edges"]
    assert ("presidency", "ts_1956", "ends_at") in matched["temporal_edges"]
    assert ("person", "ts_1950", "born_at") in matched["temporal_edges"]


def test_filter_hybrid_relabels_timestamp_bullets():

    nodes, edges = _temporal_graph()
    matches = match_temporal_neighborhood(
        build_temporal_index(nodes, edges), _utc(1950, 1, 1), _utc(1951, 1, 1)
    )
    candidates = {
        "chunks": [{"id": "c2", "text": "in 1950"}],
        "chunk_summaries": {},
        "entities": [
            {
                "id": "e2",
                "description": "Ada",
                "edges": [
                    {
                        "source": "Ada",
                        "source_id": "e2",
                        "target": "ts_1950",
                        "target_id": "ts_1950",
                        "relationship": "born_at",
                        "text": "Ada -- born_at -- ts_1950",
                    }
                ],
            }
        ],
        "facts": [],
    }

    result = filter_hybrid(candidates, matches, top_k=5)

    [bullet] = result["entities"][0]["edges"]
    assert bullet["target"] == "1950"
    assert bullet["text"] == "Ada -- born_at -- 1950"


def _candidates():
    entity_temporal = {
        "id": "e2",
        "description": "keep me",
        "edges": [
            {"source_id": "e2", "target_id": "ts_1950", "relationship": "born_at"},
            {"source_id": "e2", "target_id": "x", "relationship": "works_at"},
        ],
    }
    return {
        "chunks": [{"id": "c1", "text": "unrelated"}, {"id": "c2", "text": "in 1950"}],
        "chunk_summaries": {"c1": "s1", "c2": "s2"},
        "entities": [{"id": "e1", "description": "other", "edges": []}, entity_temporal],
        "facts": ["f1", "f2"],
    }


def _temporal_graph():
    nodes = [
        _graph_node("c1", "DocumentChunk"),
        _graph_node("c2", "DocumentChunk"),
        _graph_node("ts_1950", "Timestamp", timestamp_str="1950"),
        _graph_node("e2", "Entity"),
    ]
    edges = [
        _graph_edge("c2", "ts_1950", "contains"),
        _graph_edge("c2", "e2", "contains"),
        _graph_edge("e2", "ts_1950", "born_at"),
    ]
    return nodes, edges


def _retriever(monkeypatch, *, nodes, edges, interval, candidates, empty=False):

    class FakeGraphEngine:
        calls = 0

        async def get_graph_data(self):
            FakeGraphEngine.calls += 1
            return nodes, edges

    class FakeGraph:
        async def is_empty(self):
            return empty

    class FakeUnified:
        graph = FakeGraph()

    async def fake_graph_engine():
        return FakeGraphEngine()

    async def fake_unified_engine():
        return FakeUnified()

    hybrid_fetch = AsyncMock(return_value=candidates)
    extract = AsyncMock(return_value=interval)
    monkeypatch.setattr(
        "cognee.modules.retrieval.temporal_hybrid_retriever.get_graph_engine", fake_graph_engine
    )
    monkeypatch.setattr(
        "cognee.modules.retrieval.temporal_hybrid_retriever.get_unified_engine", fake_unified_engine
    )
    monkeypatch.setattr(
        "cognee.modules.retrieval.temporal_hybrid_retriever.extract_query_interval", extract
    )
    monkeypatch.setattr(
        "cognee.modules.retrieval.temporal_hybrid_retriever.HybridRetriever.get_retrieved_objects",
        hybrid_fetch,
    )
    retriever = TemporalHybridRetriever(candidate_top_k=20, top_k=1)
    return retriever, FakeGraphEngine, hybrid_fetch, extract


def test_temporal_retriever_rejects_bad_limits():

    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=1, top_k=2)
    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=5, top_k=0)


@pytest.mark.asyncio
async def test_temporal_retriever_empty_graph_makes_no_external_calls(monkeypatch):
    retriever, _engine, hybrid_fetch, extract = _retriever(
        monkeypatch, nodes=[], edges=[], interval=None, candidates=None, empty=True
    )
    with pytest.raises(ValueError, match="blank"):
        await retriever.get_retrieved_objects(query="   ")

    result = await retriever.get_retrieved_objects(query="in 1950")
    assert retriever.last_reason == "empty_graph"
    assert result["chunks"] == []
    hybrid_fetch.assert_not_called()
    extract.assert_not_called()


@pytest.mark.asyncio
async def test_temporal_retriever_filters_before_limit_and_restricts_entities(monkeypatch):
    nodes, edges = _temporal_graph()
    candidates = _candidates()
    retriever, engine, _fetch, _extract = _retriever(
        monkeypatch,
        nodes=nodes,
        edges=edges,
        interval=(_utc(1950, 1, 1), _utc(1951, 1, 1), None),
        candidates=candidates,
    )
    result = await retriever.get_retrieved_objects(query="in 1950")

    assert retriever.last_reason is None
    assert [chunk["id"] for chunk in result["chunks"]] == ["c2"]
    assert result["chunk_summaries"] == {"c2": "s2"}
    assert result["facts"] == []
    filtered_entity = result["entities"][0]
    assert filtered_entity["description"] is None
    assert filtered_entity["edges"] == [
        {
            "source_id": "e2",
            "target_id": "ts_1950",
            "relationship": "born_at",
            "target": "1950",
            "text": "e2 -- born_at -- 1950",
        }
    ]

    baseline = retriever.last_baseline
    assert [chunk["id"] for chunk in baseline["chunks"]] == ["c1"]
    assert baseline["facts"] == ["f1"]
    assert baseline["entities"][0]["description"] == "other"
    assert candidates["entities"][1]["description"] == "keep me"
    assert len(candidates["entities"][1]["edges"]) == 2

    await retriever.get_retrieved_objects(query="in 1950")
    assert engine.calls == 1  # the temporal index is built once per instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interval", "graph", "reason"),
    [
        ((None, None, "no_time_constraint"), _temporal_graph(), "no_time_constraint"),
        ((None, None, "invalid_interval"), _temporal_graph(), "invalid_interval"),
        ((_utc(1800, 1, 1), _utc(1801, 1, 1), None), _temporal_graph(), "no_temporal_match"),
        (
            (_utc(1950, 1, 1), _utc(1951, 1, 1), None),
            (
                [
                    _graph_node("elsewhere", "DocumentChunk"),
                    _graph_node("ts_1950", "Timestamp", timestamp_str="1950"),
                ],
                [_graph_edge("elsewhere", "ts_1950", "contains")],
            ),
            "no_candidate_overlap",
        ),
    ],
)
async def test_temporal_retriever_fallback_reasons(interval, graph, reason, monkeypatch):
    nodes, edges = graph
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch, nodes=nodes, edges=edges, interval=interval, candidates=_candidates()
    )
    result = await retriever.get_retrieved_objects(query="query")
    assert retriever.last_reason == reason
    assert result is retriever.last_baseline
    assert [chunk["id"] for chunk in result["chunks"]] == ["c1"]


@pytest.mark.asyncio
async def test_temporal_retriever_propagates_hybrid_errors(monkeypatch):
    nodes, edges = _temporal_graph()
    retriever, _engine, hybrid_fetch, _extract = _retriever(
        monkeypatch,
        nodes=nodes,
        edges=edges,
        interval=(None, None, "no_time_constraint"),
        candidates=None,
    )
    hybrid_fetch.side_effect = RuntimeError("hybrid")
    with pytest.raises(RuntimeError, match="hybrid"):
        await retriever.get_retrieved_objects(query="in 1950")
