"""TemporalHybridRetriever: interval extraction, anchors, rerank, fallbacks (SDK-828)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.temporal_hybrid.matching import (
    UNDATED_NOTE,
    anchors_from_neighborhood,
    chunks_containing,
    empty_anchors,
    extract_query_interval,
    passage_notes,
    rerank_hybrid,
    slice_hybrid,
    to_epoch_ms,
    window_preamble,
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
    return {"year": year, "month": month, "day": day, "hour": 0, "minute": 0, "second": 0}


# --- interval extraction ---------------------------------------------------


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


def test_to_epoch_ms():
    assert to_epoch_ms(None) is None
    assert to_epoch_ms(_utc(1970, 1, 1, 0, 0, 1)) == 1000


# --- anchors -----------------------------------------------------------------


def _node(node_id: str, type_name: str, **properties):
    return (node_id, {"type": type_name, **properties})


def _edge(source: str, target: str, relationship: str):
    return (source, target, relationship, {})


def test_anchors_come_from_edges_into_the_matched_timestamps():
    nodes = [
        _node("c1", "DocumentChunk"),
        _node("c2", "DocumentChunk"),
        _node("ts_1950", "Timestamp", timestamp_str="1950"),
        _node("ts_1960", "Timestamp", timestamp_str="1960"),
        _node("ada", "Entity"),
        _node("acme", "Entity"),
        _node("person", "EntityType"),
    ]
    edges = [
        _edge("c1", "ts_1950", "contains"),
        _edge("c2", "ts_1960", "contains"),
        _edge("ada", "ts_1950", "born_at"),
        _edge("acme", "ts_1960", "founded_at"),
        _edge("ada", "person", "is_a"),
        _edge("c1", "ada", "contains"),
    ]
    anchors = anchors_from_neighborhood({"ts_1950"}, nodes, edges)

    assert anchors["chunk_ids"] == {"c1"}
    assert anchors["entity_ids"] == {"ada"}  # any edge into the timestamp, not only *_at


def test_chunks_containing_follows_contains_edges_only_from_chunks():
    nodes = [
        _node("c1", "DocumentChunk"),
        _node("c2", "DocumentChunk"),
        _node("ada", "Entity"),
        _node("x", "Entity"),
    ]
    edges = [
        _edge("c1", "ada", "contains"),
        _edge("x", "ada", "knows"),
        _edge("c2", "x", "contains"),
    ]

    assert chunks_containing({"ada"}, nodes, edges) == {"c1": {"ada"}}


# --- rerank ----------------------------------------------------------------------


def _candidates():
    return {
        "chunks": [
            {"id": "c1", "text": "unrelated"},
            {"id": "c2", "text": "in 1950"},
            {"id": "c3", "text": "also 1950"},
        ],
        "chunk_summaries": {"c1": "s1", "c2": "s2", "c3": "s3"},
        "entities": [
            {"id": "e1", "description": "other", "edges": []},
            {
                "id": "e2",
                "description": "keep me",
                "edges": [{"relationship": "born_at"}, {"relationship": "works_at"}],
            },
        ],
        "facts": ["f1", "f2"],
    }


def test_rerank_puts_anchored_candidates_first_and_keeps_the_rest():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"c3", "c2"}, "entity_ids": {"e2"}}
    result = rerank_hybrid(_candidates(), anchors, top_k=2)

    assert [chunk["id"] for chunk in result["chunks"]] == [
        "c2",
        "c3",
    ]  # hybrid order among anchored
    assert result["chunk_summaries"] == {"c2": "s2", "c3": "s3"}
    assert [entity["id"] for entity in result["entities"]] == ["e2", "e1"]
    assert result["entities"][0]["description"] == "keep me"  # nothing stripped
    assert len(result["entities"][0]["edges"]) == 2
    assert result["facts"] == ["f1", "f2"]


def test_rerank_with_no_anchored_candidate_is_the_plain_slice():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"elsewhere"}, "entity_ids": set()}
    assert rerank_hybrid(_candidates(), anchors, top_k=2) == slice_hybrid(_candidates(), 2)


def test_rerank_fills_up_with_unanchored_candidates():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"c3"}, "entity_ids": set()}
    result = rerank_hybrid(_candidates(), anchors, top_k=2)
    assert [chunk["id"] for chunk in result["chunks"]] == ["c3", "c1"]


# --- retriever flow ------------------------------------------------------------


def _retriever(monkeypatch, *, timestamps, neighborhoods, interval, candidates, empty=False):
    class FakeGraphEngine:
        range_calls: list = []
        neighborhood_calls: list = []

        async def get_timestamps_in_range(self, start, end):
            FakeGraphEngine.range_calls.append((start, end))
            return timestamps

        async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
            FakeGraphEngine.neighborhood_calls.append((list(node_ids), depth, edge_types))
            return neighborhoods.get(tuple(node_ids), ([], []))

    class FakeGraph:
        async def is_empty(self):
            return empty

    class FakeUnified:
        graph = FakeGraph()

    async def fake_graph_engine():
        return FakeGraphEngine()

    async def fake_unified_engine():
        return FakeUnified()

    FakeGraphEngine.range_calls = []
    FakeGraphEngine.neighborhood_calls = []
    hybrid_fetch = AsyncMock(return_value=candidates)
    extract = AsyncMock(return_value=interval)
    module = "cognee.modules.retrieval.temporal_hybrid_retriever."
    monkeypatch.setattr(module + "get_graph_engine", fake_graph_engine)
    monkeypatch.setattr(module + "get_unified_engine", fake_unified_engine)
    monkeypatch.setattr(module + "extract_query_interval", extract)
    monkeypatch.setattr(module + "HybridRetriever.get_retrieved_objects", hybrid_fetch)
    retriever = TemporalHybridRetriever(candidate_top_k=20, top_k=2)
    return retriever, FakeGraphEngine, hybrid_fetch, extract


def test_temporal_retriever_rejects_bad_limits():
    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=1, top_k=2)
    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=5, top_k=0)
    assert TemporalHybridRetriever(top_k=3).chunks_top_k == 12


@pytest.mark.asyncio
async def test_temporal_retriever_empty_graph_makes_no_external_calls(monkeypatch):
    retriever, engine, hybrid_fetch, extract = _retriever(
        monkeypatch, timestamps=[], neighborhoods={}, interval=None, candidates=None, empty=True
    )
    with pytest.raises(ValueError, match="blank"):
        await retriever.get_retrieved_objects(query="   ")

    result = await retriever.get_retrieved_objects(query="in 1950")
    assert retriever.last_reason == "empty_graph"
    assert result["chunks"] == []
    hybrid_fetch.assert_not_called()
    extract.assert_not_called()
    assert engine.range_calls == []


@pytest.mark.asyncio
async def test_temporal_retriever_reranks_by_timestamp_and_entity_anchors(monkeypatch):
    ts_neighborhood = (
        [_node("c2", "DocumentChunk"), _node("ts_1950", "Timestamp"), _node("e2", "Entity")],
        [_edge("c2", "ts_1950", "contains"), _edge("e2", "ts_1950", "born_at")],
    )
    entity_neighborhood = (
        [_node("c3", "DocumentChunk"), _node("c2", "DocumentChunk"), _node("e2", "Entity")],
        [_edge("c3", "e2", "contains"), _edge("c2", "e2", "contains")],
    )
    retriever, engine, _fetch, _extract = _retriever(
        monkeypatch,
        timestamps=[{"id": "ts_1950", "timestamp_str": "1950", "time_at": 0, "time_until": 1}],
        neighborhoods={("ts_1950",): ts_neighborhood, ("e2",): entity_neighborhood},
        interval=(_utc(1950, 1, 1), _utc(1951, 1, 1), None),
        candidates=_candidates(),
    )
    result = await retriever.get_retrieved_objects(query="in 1950")

    assert retriever.last_reason is None
    assert engine.range_calls == [(to_epoch_ms(_utc(1950, 1, 1)), to_epoch_ms(_utc(1951, 1, 1)))]
    assert engine.neighborhood_calls == [(["ts_1950"], 1, None), (["e2"], 1, ["contains"])]
    assert retriever.last_anchors["chunk_ids"] == {"c2", "c3"}  # c3 via the entity anchor
    assert retriever.last_anchors["entity_ids"] == {"e2"}
    assert [chunk["id"] for chunk in result["chunks"]] == ["c2", "c3"]
    assert [entity["id"] for entity in result["entities"]] == ["e2", "e1"]
    assert result["entities"][0]["description"] == "keep me"
    assert [chunk["id"] for chunk in retriever.last_baseline["chunks"]] == ["c1", "c2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interval", "timestamps", "reason"),
    [
        ((None, None, "no_time_constraint"), [], "no_time_constraint"),
        ((None, None, "invalid_interval"), [], "invalid_interval"),
        ((_utc(1800, 1, 1), _utc(1801, 1, 1), None), [], "no_temporal_match"),
        (
            (_utc(1950, 1, 1), _utc(1951, 1, 1), None),
            [{"id": "ts_elsewhere", "timestamp_str": "1950", "time_at": 0, "time_until": 1}],
            "no_candidate_overlap",
        ),
    ],
)
async def test_temporal_retriever_fallbacks_return_the_baseline(
    monkeypatch, interval, timestamps, reason
):
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        timestamps=timestamps,
        neighborhoods={
            ("ts_elsewhere",): (
                [_node("far", "DocumentChunk")],
                [_edge("far", "ts_elsewhere", "contains")],
            )
        },
        interval=interval,
        candidates=_candidates(),
    )
    result = await retriever.get_retrieved_objects(query="in 1950")

    assert retriever.last_reason == reason
    assert result == retriever.last_baseline
    assert [chunk["id"] for chunk in result["chunks"]] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_temporal_retriever_propagates_hybrid_errors(monkeypatch):
    retriever, _engine, hybrid_fetch, _extract = _retriever(
        monkeypatch, timestamps=[], neighborhoods={}, interval=(None, None, None), candidates=None
    )
    hybrid_fetch.side_effect = RuntimeError("hybrid down")
    with pytest.raises(RuntimeError, match="hybrid down"):
        await retriever.get_retrieved_objects(query="in 1950")


# --- context notes: what the model is told about the window -------------------


def test_anchors_keep_the_matched_dates_per_chunk_and_entity():
    nodes = [
        _node("c1", "DocumentChunk"),
        _node("ts", "Timestamp", timestamp_str="1950"),
        _node("ada", "Entity", name="Ada Lovelace"),
    ]
    edges = [_edge("c1", "ts", "contains"), _edge("ada", "ts", "born_at")]

    anchors = anchors_from_neighborhood({"ts"}, nodes, edges)

    assert anchors["chunk_times"] == {"c1": {"1950"}}
    assert anchors["entity_times"] == {"ada": {"1950"}}
    assert anchors["entity_names"] == {"ada": "Ada Lovelace"}


def test_passage_notes_name_own_dates_inherited_dates_or_the_absence():
    anchors = {
        **empty_anchors(),
        "chunk_times": {"c1": {"1950-03", "1950"}},
        "chunk_via": {"c2": {"ada"}},
        "entity_times": {"ada": {"1950"}},
        "entity_names": {"ada": "Ada Lovelace"},
    }
    chunks = [{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}, {"id": "c3", "text": "c"}]

    assert passage_notes(chunks, anchors) == {
        "c1": "time: 1950, 1950-03",
        "c2": "time: 1950 (through Ada Lovelace)",
        "c3": UNDATED_NOTE,
    }


@pytest.mark.parametrize(
    ("start", "end", "period"),
    [
        (_utc(1950, 1, 1), _utc(1951, 1, 1), "1950-01-01 to 1951-01-01"),
        (None, _utc(1900, 1, 1), "before 1900-01-01"),
        (_utc(1969, 7, 20, 20, 17), None, "from 1969-07-20 20:17:00 onward"),
    ],
)
def test_window_preamble_states_the_period(start, end, period):
    text = window_preamble(start, end, anchored=True)
    assert text.startswith("## Time window\n")
    assert f"Question period: {period} (UTC, end exclusive)." in text
    assert "No passage or entity" not in text
    assert "No passage or entity in this context is dated inside this period." in (
        window_preamble(start, end, anchored=False)
    )


@pytest.mark.asyncio
async def test_temporal_context_marks_every_passage(monkeypatch):
    ts_neighborhood = (
        [
            _node("c2", "DocumentChunk"),
            _node("ts_1950", "Timestamp", timestamp_str="1950"),
            _node("e2", "Entity", name="Ada"),
        ],
        [_edge("c2", "ts_1950", "contains"), _edge("e2", "ts_1950", "born_at")],
    )
    entity_neighborhood = (
        [_node("c3", "DocumentChunk"), _node("e2", "Entity", name="Ada")],
        [_edge("c3", "e2", "contains")],
    )
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        timestamps=[{"id": "ts_1950", "timestamp_str": "1950", "time_at": 0, "time_until": 1}],
        neighborhoods={("ts_1950",): ts_neighborhood, ("e2",): entity_neighborhood},
        interval=(_utc(1950, 1, 1), _utc(1951, 1, 1), None),
        candidates=_candidates(),
    )
    retriever.top_k = 3
    result = await retriever.get_retrieved_objects(query="in 1950")
    context = await retriever.get_context_from_objects(
        query="in 1950", retrieved_objects={**result, "facts": []}
    )

    assert context.startswith("## Time window\nQuestion period: 1950-01-01 to 1951-01-01")
    assert "## Relevant passages\ntime: 1950\nin 1950\n---\n" in context
    assert "time: 1950 (through Ada)\nalso 1950\n---\n" in context
    assert f"{UNDATED_NOTE}\nunrelated" in context
    assert "No passage or entity" not in context


@pytest.mark.asyncio
async def test_temporal_context_says_when_nothing_in_the_window_matched(monkeypatch):
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        timestamps=[],
        neighborhoods={},
        interval=(_utc(1800, 1, 1), _utc(1801, 1, 1), None),
        candidates=_candidates(),
    )
    result = await retriever.get_retrieved_objects(query="in 1800")
    context = await retriever.get_context_from_objects(
        query="in 1800", retrieved_objects={**result, "facts": []}
    )

    assert "No passage or entity in this context is dated inside this period." in context
    assert context.count(UNDATED_NOTE) == 1 + len(result["chunks"])  # preamble + each passage


@pytest.mark.asyncio
async def test_temporal_context_without_a_window_is_the_plain_hybrid_one(monkeypatch):
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        timestamps=[],
        neighborhoods={},
        interval=(None, None, "no_time_constraint"),
        candidates=_candidates(),
    )
    result = await retriever.get_retrieved_objects(query="who?")
    context = await retriever.get_context_from_objects(
        query="who?", retrieved_objects={**result, "facts": []}
    )

    assert "Time window" not in context
    assert "time:" not in context
    assert context.startswith("## Relevant passages\nunrelated")
