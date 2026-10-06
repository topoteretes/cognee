"""TemporalHybridRetriever: interval extraction, anchors, rerank, fallbacks (SDK-828)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.hybrid.candidates import HybridCandidates
from cognee.modules.retrieval.hybrid.facts import FactCandidates
from cognee.modules.retrieval.temporal_hybrid.matching import (
    extract_query_interval,
    to_epoch_ms,
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


def prioritized(candidates: HybridCandidates, anchors: dict) -> HybridCandidates:
    return candidates.prioritize(anchors["chunk_ids"], anchors["entity_ids"])


# --- prioritize + finalize ------------------------------------------------

HITS = [
    {"id": "f_atlas", "text": "Atlas was founded in 1950"},
    {"id": "f_helios", "text": "Helios launched in 1898"},
    {"id": "f_other", "text": "Something else entirely happened"},
]


def _candidates(fact_candidates: FactCandidates | None = None) -> HybridCandidates:
    return HybridCandidates(
        chunks=[
            {"id": "c1", "text": "unrelated"},
            {"id": "c2", "text": "in 1950"},
            {"id": "c3", "text": "also 1950"},
        ],
        chunk_summaries={"c1": "s1", "c2": "s2", "c3": "s3"},
        entities=[
            {"id": "e1", "description": "other", "edges": []},
            {
                "id": "e2",
                "description": "keep me",
                "edges": [{"relationship": "born_at"}, {"relationship": "works_at"}],
            },
        ],
        fact_candidates=fact_candidates or FactCandidates(edge_hits=HITS, facts_top_k=2),
    )


def _finalize(candidates: HybridCandidates, top_k: int = 2) -> dict:
    return candidates.finalize(chunks_limit=top_k, entities_limit=top_k)


def test_prioritize_puts_anchored_candidates_first_and_keeps_the_rest():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"c3", "c2"}, "entity_ids": {"e2"}}
    result = _finalize(prioritized(_candidates(), anchors))

    # hybrid order among the anchored, then the rest; nothing stripped from a candidate
    assert [chunk["id"] for chunk in result["chunks"]] == ["c2", "c3"]
    assert result["chunk_summaries"] == {"c2": "s2", "c3": "s3"}
    assert [entity["id"] for entity in result["entities"]] == ["e2", "e1"]
    assert result["entities"][0]["description"] == "keep me"
    assert len(result["entities"][0]["edges"]) == 2
    # facts are selected after the cut, against the entities kept (none carry these hits)
    assert [fact["id"] for fact in result["facts"]] == ["f_atlas", "f_helios"]


def test_prioritize_with_no_anchored_candidate_is_the_plain_slice():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"elsewhere"}, "entity_ids": set()}
    assert _finalize(prioritized(_candidates(), anchors)) == _finalize(_candidates())


def test_prioritize_fills_up_with_unanchored_candidates():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"c3"}, "entity_ids": set()}
    result = _finalize(prioritized(_candidates(), anchors))
    assert [chunk["id"] for chunk in result["chunks"]] == ["c3", "c1"]


def test_prioritize_drops_nothing():
    anchors = {"timestamp_ids": {"ts"}, "chunk_ids": {"c3"}, "entity_ids": {"e2"}}
    reordered = prioritized(_candidates(), anchors)
    assert [chunk["id"] for chunk in reordered.chunks] == ["c3", "c1", "c2"]
    assert [entity["id"] for entity in reordered.entities] == ["e2", "e1"]
    assert reordered.chunk_summaries == _candidates().chunk_summaries
    assert reordered.fact_candidates == _candidates().fact_candidates


def test_finalize_selects_facts_against_the_entities_it_keeps():
    """A fact deduplicated against a candidate entity must not vanish with that entity."""
    candidates = HybridCandidates(
        chunks=[{"id": "c1", "text": "x"}],
        entities=[
            {"id": "atlas", "description": "a", "edges": [{"edge_type_id": "f_atlas"}]},
            {"id": "helios", "description": "h", "edges": [{"edge_type_id": "f_helios"}]},
        ],
        fact_candidates=FactCandidates(edge_hits=HITS, facts_top_k=2),
    )
    # Both shown: both facts are bullets already, only the unrelated one stands alone.
    both = _finalize(candidates, top_k=2)
    assert [fact["id"] for fact in both["facts"]] == ["f_other"]
    # Helios cut: its fact is no longer shown under an entity and comes back standalone,
    # ahead of the unrelated one, in hit order; Atlas' stays deduplicated.
    one = _finalize(candidates, top_k=1)
    assert [entity["id"] for entity in one["entities"]] == ["atlas"]
    assert [fact["id"] for fact in one["facts"]] == ["f_helios", "f_other"]


def test_finalize_can_size_the_fallback_fact_budget_for_the_entities_it_shows():
    """With no entity kept, facts get the entity lane's edge budget. The fetch sized
    it for its own limit; a caller cutting to fewer entities passes its own."""
    hits = [{"id": f"f{i}", "text": f"fact number {i} happened"} for i in range(100)]
    candidates = HybridCandidates(
        chunks=[{"id": "c1"}],
        fact_candidates=FactCandidates(edge_hits=hits, facts_top_k=2, entity_edge_budget=60),
    )
    as_fetched = candidates.finalize(chunks_limit=2, entities_limit=2)
    assert len(as_fetched["facts"]) == 60  # the fetch's budget, unchanged by default
    shown_two = candidates.finalize(chunks_limit=2, entities_limit=2, entity_edge_budget=6)
    assert len(shown_two["facts"]) == 6


# --- retriever flow ------------------------------------------------------------


def _retriever(monkeypatch, *, anchors, interval, candidates, timestamps=(), empty=False):
    class FakeGraphEngine:
        anchor_calls: list = []
        range_calls: list = []

        async def get_temporal_anchors(self, chunk_ids, entity_ids, start, end):
            FakeGraphEngine.anchor_calls.append((set(chunk_ids), set(entity_ids), start, end))
            return anchors

        async def get_timestamps_in_range(self, start, end):
            FakeGraphEngine.range_calls.append((start, end))
            return list(timestamps)

    class FakeGraph:
        async def is_empty(self):
            return empty

    class FakeUnified:
        graph = FakeGraph()

    async def fake_graph_engine():
        return FakeGraphEngine()

    async def fake_unified_engine():
        return FakeUnified()

    FakeGraphEngine.anchor_calls = []
    FakeGraphEngine.range_calls = []
    retriever = TemporalHybridRetriever(candidate_top_k=20, top_k=2)
    hybrid_fetch = AsyncMock(return_value=candidates)
    extract = AsyncMock(return_value=interval)
    module = "cognee.modules.retrieval.temporal_hybrid_retriever."
    monkeypatch.setattr(module + "get_graph_engine", fake_graph_engine)
    monkeypatch.setattr(module + "get_unified_engine", fake_unified_engine)
    monkeypatch.setattr(module + "extract_query_interval", extract)
    monkeypatch.setattr(module + "HybridRetriever._fetch_candidates", hybrid_fetch)
    return retriever, FakeGraphEngine, hybrid_fetch, extract


def _anchors(chunks=(), entities=(), timestamps=("ts_1950",)) -> dict:
    return {
        "timestamp_ids": set(timestamps),
        "chunk_ids": set(chunks),
        "entity_ids": set(entities),
    }


def test_temporal_retriever_rejects_bad_limits():
    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=1, top_k=2)
    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=5, top_k=0)
    assert TemporalHybridRetriever(top_k=3).chunks_top_k == 12
    # A REST request may carry ``top_k: null``; it must resolve, not multiply None.
    assert TemporalHybridRetriever(top_k=None).top_k == 5
    assert TemporalHybridRetriever(top_k=None).chunks_top_k == 20


@pytest.mark.asyncio
async def test_temporal_retriever_empty_graph_makes_no_external_calls(monkeypatch):
    retriever, engine, hybrid_fetch, extract = _retriever(
        monkeypatch, anchors=_anchors(), interval=None, candidates=None, empty=True
    )
    with pytest.raises(ValueError, match="blank"):
        await retriever.get_retrieved_objects(query="   ")

    result = await retriever.get_retrieved_objects(query="in 1950")
    assert retriever.last_reason == "empty_graph"
    assert result["chunks"] == []
    hybrid_fetch.assert_not_called()
    extract.assert_not_called()
    assert engine.anchor_calls == []


@pytest.mark.asyncio
async def test_temporal_retriever_reranks_by_the_candidates_anchors(monkeypatch):
    retriever, engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(chunks=("c2", "c3"), entities=("e2",)),
        interval=(_utc(1950, 1, 1), _utc(1951, 1, 1), None),
        candidates=_candidates(),
    )
    result = await retriever.get_retrieved_objects(query="in 1950")

    assert retriever.last_reason is None
    # The adapter is asked about the candidates, never about the whole window.
    assert engine.anchor_calls == [
        (
            {"c1", "c2", "c3"},
            {"e1", "e2"},
            to_epoch_ms(_utc(1950, 1, 1)),
            to_epoch_ms(_utc(1951, 1, 1)),
        )
    ]
    assert engine.range_calls == []
    assert retriever.last_anchors["chunk_ids"] == {"c2", "c3"}
    assert retriever.last_anchors["entity_ids"] == {"e2"}
    assert [chunk["id"] for chunk in result["chunks"]] == ["c2", "c3"]
    assert [entity["id"] for entity in result["entities"]] == ["e2", "e1"]
    assert result["entities"][0]["description"] == "keep me"
    assert [chunk["id"] for chunk in retriever.last_baseline["chunks"]] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_temporal_retriever_selects_facts_against_the_entities_it_shows(monkeypatch):
    """End to end through the retriever: the fetch decides no facts; the cut does."""
    candidates = HybridCandidates(
        chunks=[{"id": "c1", "text": "x"}],
        entities=[
            {"id": "atlas", "description": "a", "edges": [{"edge_type_id": "f_atlas"}]},
            {"id": "helios", "description": "h", "edges": [{"edge_type_id": "f_helios"}]},
        ],
        fact_candidates=FactCandidates(edge_hits=HITS, facts_top_k=2),
    )
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(chunks=("c1",), entities=("atlas",)),
        interval=(_utc(1950, 1, 1), _utc(1951, 1, 1), None),
        candidates=candidates,
    )
    retriever.top_k = 1

    result = await retriever.get_retrieved_objects(query="in 1950")

    assert [entity["id"] for entity in result["entities"]] == ["atlas"]  # helios cut
    assert [fact["id"] for fact in result["facts"]] == ["f_helios", "f_other"]
    assert [fact["id"] for fact in retriever.last_baseline["facts"]] == ["f_helios", "f_other"]


@pytest.mark.asyncio
async def test_temporal_retriever_caps_facts_by_top_k_when_no_entity_is_shown(monkeypatch):
    """The no-entity fact budget follows top_k, not the 4x candidate fetch."""
    hits = [{"id": f"f{i}", "text": f"fact number {i} happened"} for i in range(100)]
    candidates = HybridCandidates(
        chunks=[{"id": "c1"}],
        # what HybridRetriever._retrieve_entities_and_facts sets: candidate_top_k (20) x 3
        fact_candidates=FactCandidates(edge_hits=hits, facts_top_k=2, entity_edge_budget=60),
    )
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(chunks=("c1",)),
        interval=(_utc(1950, 1, 1), _utc(1951, 1, 1), None),
        candidates=candidates,
    )
    retriever.max_edges_per_entity = 3

    result = await retriever.get_retrieved_objects(query="in 1950")

    assert result["entities"] == []
    assert len(result["facts"]) == retriever.top_k * 3  # 6, not 60
    assert len(retriever.last_baseline["facts"]) == retriever.top_k * 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interval", "anchors", "timestamps", "reason"),
    [
        ((None, None, "no_time_constraint"), _anchors(), [], "no_time_constraint"),
        ((None, None, "invalid_interval"), _anchors(), [], "invalid_interval"),
        # no candidate is dated in the window — and the graph has no such time at all
        (
            (_utc(1800, 1, 1), _utc(1801, 1, 1), None),
            _anchors(timestamps=()),
            [],
            "no_temporal_match",
        ),
        # ... or the window's times belong to things outside the candidate set
        (
            (_utc(1950, 1, 1), _utc(1951, 1, 1), None),
            _anchors(timestamps=()),
            [{"id": "ts_elsewhere", "timestamp_str": "1950", "time_at": 0, "time_until": 1}],
            "no_candidate_overlap",
        ),
        # an anchored candidate that was already first changes nothing
        (
            (_utc(1950, 1, 1), _utc(1951, 1, 1), None),
            _anchors(chunks=("c1",)),
            [],
            "no_candidate_overlap",
        ),
    ],
)
async def test_temporal_retriever_fallbacks_return_the_baseline(
    monkeypatch, interval, anchors, timestamps, reason
):
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=anchors,
        timestamps=timestamps,
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
        monkeypatch, anchors=_anchors(), interval=(None, None, None), candidates=None
    )
    hybrid_fetch.side_effect = RuntimeError("hybrid down")
    with pytest.raises(RuntimeError, match="hybrid down"):
        await retriever.get_retrieved_objects(query="in 1950")
