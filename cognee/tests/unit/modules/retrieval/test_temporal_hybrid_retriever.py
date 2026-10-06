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


def test_prioritize_orders_named_chunks_by_rank_and_keeps_the_rest():
    """c1..c3 anchored; the rank puts c3 first, then c1; c2 has no rank and goes last among
    the anchored. Nothing is dropped and entities are untouched."""
    reordered = _candidates().prioritize(
        {"c1", "c2", "c3"}, set(), chunk_rank={"c3": 1.0, "c1": 5.0}
    )
    assert [chunk["id"] for chunk in reordered.chunks] == ["c3", "c1", "c2"]
    flat = _candidates().prioritize({"c1", "c2", "c3"}, set())
    assert [chunk["id"] for chunk in flat.chunks] == ["c1", "c2", "c3"]


def test_tightest_first_orders_by_precision_then_time():
    from cognee.modules.retrieval.temporal_hybrid.matching import tightest_first

    day = 86_400_000
    nodes = [
        {"id": "year", "time_at": 0, "time_until": 365 * day},
        {"id": "day_late", "time_at": 200 * day, "time_until": 201 * day},
        {"id": "day_early", "time_at": 10 * day, "time_until": 11 * day},
        {"id": "month", "time_at": 0, "time_until": 31 * day},
    ]
    assert [n["id"] for n in tightest_first(nodes)] == ["day_early", "day_late", "month", "year"]


def test_tightness_rank_prefers_the_timestamp_that_fits_the_window():
    from cognee.modules.retrieval.temporal_hybrid.matching import tightness_rank

    day, year = 86_400_000, 365 * 86_400_000
    in_window = [
        {"id": "ts_day", "time_at": 0, "time_until": day},
        {"id": "ts_year", "time_at": -(100 * day), "time_until": -(100 * day) + year},
        {"id": "ts_legacy", "time_at": 0},  # no time_until: a second
    ]
    rank = tightness_rank(
        {
            "c_day": {"ts_day", "ts_year"},
            "c_year": {"ts_year"},
            "c_legacy": {"ts_legacy"},
            "c_none": {"ts_x"},
        },
        in_window,
    )
    assert rank["c_legacy"] < rank["c_day"] < rank["c_year"]
    assert rank["c_none"] == float("inf")


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


def _retriever(
    monkeypatch,
    *,
    anchors,
    interval,
    candidates,
    timestamps=(),
    empty=False,
    neighborhood=None,
    vector_rows=None,
):
    """``neighborhood`` is what get_neighborhood returns for the in-window timestamps
    (nodes, edges); ``vector_rows`` maps a collection name to the rows retrieve()
    can return, so the expansion has something to resolve attached ids against."""

    class FakeGraphEngine:
        anchor_calls: list = []
        range_calls: list = []
        neighborhood_calls: list = []

        async def get_temporal_anchors(self, chunk_ids, entity_ids, start, end):
            FakeGraphEngine.anchor_calls.append((set(chunk_ids), set(entity_ids), start, end))
            return anchors

        async def get_timestamps_in_range(self, start, end):
            FakeGraphEngine.range_calls.append((start, end))
            return list(timestamps)

        async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
            FakeGraphEngine.neighborhood_calls.append(list(node_ids))
            if neighborhood is None:
                return [], []
            nodes, edges = neighborhood
            seeds = set(node_ids)
            # Serve both the expansion (timestamp seeds) and build_entities (entity seeds).
            return nodes, [edge for edge in edges if edge[0] in seeds or edge[1] in seeds]

    class FakeGraph:
        async def is_empty(self):
            return empty

    class FakeVector:
        async def has_collection(self, name):
            return name in (vector_rows or {})

        async def retrieve(self, collection, ids):
            wanted = set(ids)
            return [row for row in (vector_rows or {}).get(collection, []) if row["id"] in wanted]

    class FakeUnified:
        graph = FakeGraph()
        vector = FakeVector()

    async def fake_graph_engine():
        return FakeGraphEngine()

    async def fake_unified_engine():
        return FakeUnified()

    FakeGraphEngine.anchor_calls = []
    FakeGraphEngine.range_calls = []
    FakeGraphEngine.neighborhood_calls = []
    retriever = TemporalHybridRetriever(candidate_top_k=20, top_k=2)
    hybrid_fetch = AsyncMock(return_value=candidates)
    extract = AsyncMock(return_value=interval)
    module = "cognee.modules.retrieval.temporal_hybrid_retriever."
    monkeypatch.setattr(module + "get_graph_engine", fake_graph_engine)
    monkeypatch.setattr(module + "get_unified_engine", fake_unified_engine)
    monkeypatch.setattr(module + "extract_query_interval", extract)
    monkeypatch.setattr(module + "HybridRetriever._fetch_candidates", hybrid_fetch)
    return retriever, FakeGraphEngine, hybrid_fetch, extract


def _anchors(chunks=(), entities=(), timestamps=("ts_1950",), chunk_timestamps=None) -> dict:
    """``chunk_timestamps`` defaults to every anchored chunk reaching the first timestamp."""
    return {
        "timestamp_ids": set(timestamps),
        "chunk_ids": set(chunks),
        "entity_ids": set(entities),
        "chunk_timestamps": (
            {chunk: {timestamps[0]} for chunk in chunks}
            if chunk_timestamps is None
            else chunk_timestamps
        ),
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
    # The window is read once; with no timestamp in it nothing is added and the
    # anchors are asked about hybrid's own candidates.
    assert engine.range_calls == [(to_epoch_ms(_utc(1950, 1, 1)), to_epoch_ms(_utc(1951, 1, 1)))]
    assert engine.neighborhood_calls == []
    assert engine.anchor_calls == [
        (
            {"c1", "c2", "c3"},
            {"e1", "e2"},
            to_epoch_ms(_utc(1950, 1, 1)),
            to_epoch_ms(_utc(1951, 1, 1)),
        )
    ]
    assert retriever.last_anchors["chunk_ids"] == {"c2", "c3"}
    assert retriever.last_anchors["entity_ids"] == {"e2"}
    assert [chunk["id"] for chunk in result["chunks"]] == ["c2", "c3"]
    assert [entity["id"] for entity in result["entities"]] == ["e2", "e1"]
    assert result["entities"][0]["description"] == "keep me"
    assert [chunk["id"] for chunk in retriever.last_baseline["chunks"]] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_temporal_retriever_shows_the_chunk_dated_to_the_day_first(monkeypatch):
    """Voskhod 2: the chunk whose timestamp is 1965-03-18 was ranked behind chunks that
    only reach the year node "1965", which also overlaps the day. Tighter fit wins."""
    day = 86_400_000
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(
            chunks=("c1", "c2", "c3"),
            timestamps=("ts_year", "ts_day"),
            chunk_timestamps={"c1": {"ts_year"}, "c2": {"ts_year"}, "c3": {"ts_day", "ts_year"}},
        ),
        interval=(_utc(1965, 3, 18), _utc(1965, 3, 19), None),
        candidates=_candidates(),
        timestamps=[
            {"id": "ts_year", "time_at": 0, "time_until": 365 * day},
            {"id": "ts_day", "time_at": 76 * day, "time_until": 77 * day},
        ],
    )
    result = await retriever.get_retrieved_objects(query="on 18 March 1965")
    assert [chunk["id"] for chunk in result["chunks"]] == ["c3", "c1"]
    assert retriever.last_anchors["chunk_timestamps"]["c3"] == {"ts_day", "ts_year"}


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
    monkeypatch, caplog, interval, anchors, timestamps, reason
):
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=anchors,
        timestamps=timestamps,
        interval=interval,
        candidates=_candidates(),
    )
    with caplog.at_level("WARNING", logger="TemporalHybridRetriever"):
        result = await retriever.get_retrieved_objects(query="in 1950")

    assert retriever.last_reason == reason
    assert result == retriever.last_baseline
    assert [chunk["id"] for chunk in result["chunks"]] == ["c1", "c2"]
    # A rerank that ran but moved nothing is not a fallback; the rest warn.
    assert ("No time-anchored data found for this question" in caplog.text) is not bool(
        anchors["chunk_ids"]
    )


@pytest.mark.asyncio
async def test_temporal_retriever_propagates_hybrid_errors(monkeypatch):
    retriever, _engine, hybrid_fetch, _extract = _retriever(
        monkeypatch, anchors=_anchors(), interval=(None, None, None), candidates=None
    )
    hybrid_fetch.side_effect = RuntimeError("hybrid down")
    with pytest.raises(RuntimeError, match="hybrid down"):
        await retriever.get_retrieved_objects(query="in 1950")


# --- window expansion ------------------------------------------------------------

TS_1950 = {"id": "ts_1950", "timestamp_str": "1950", "time_at": 0, "time_until": 1}
TS_1950_06 = {"id": "ts_1950_06", "timestamp_str": "1950-06", "time_at": 2, "time_until": 3}
WINDOW_1950 = (_utc(1950, 1, 1), _utc(1951, 1, 1), None)


def _dated_graph():
    """Rows r1..r3 and entity atlas hang off the two 1950 timestamps; c1 is a candidate
    already; a schema table also points at a timestamp but is in no collection."""
    nodes = [(node_id, {"id": node_id}) for node_id in ("r1", "r2", "r3", "atlas", "tbl", "c1")]
    edges = [
        ("r2", "ts_1950_06", "order_date", {}),
        ("r1", "ts_1950", "order_date", {}),
        ("r3", "ts_1950", "order_date", {}),
        ("c1", "ts_1950", "contains", {}),
        ("atlas", "ts_1950", "founded_at", {}),
        ("tbl", "ts_1950", "whatever", {}),
        ("atlas", "helios", "works_with", {"relationship_name": "works_with"}),
    ]
    return nodes, edges


def _rows():
    return {
        "DltRow_text": [
            {"id": "r1", "text": "row 1"},
            {"id": "r2", "text": "row 2"},
            {"id": "r3", "text": "row 3", "belongs_to_set": ["other"]},
        ],
        "DocumentChunk_text": [{"id": "c1", "text": "unrelated"}],
        "Entity_name": [{"id": "atlas", "name": "Atlas", "description": "a company"}],
    }


@pytest.mark.asyncio
async def test_window_expansion_adds_the_rows_the_window_points_at(monkeypatch):
    """Nine rows approved on one day, one in the candidates: the other eight join.
    Entities attached to the window do not: they stay with the entity lane."""
    retriever, engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(chunks=("c1", "r1", "r2", "r3"), entities=("atlas",)),
        interval=WINDOW_1950,
        candidates=_candidates(),
        timestamps=[TS_1950, TS_1950_06],
        neighborhood=_dated_graph(),
        vector_rows=_rows(),
    )
    retriever.top_k = 4
    result = await retriever.get_retrieved_objects(query="in 1950")

    assert retriever.last_reason is None
    # expansion: attached ids in timestamp order, the table (no collection) dropped,
    # c1 already a candidate; the anchors are then asked about the widened set
    assert retriever.last_expansion == {"r1", "r2", "r3"}
    assert engine.anchor_calls[0][0] == {"c1", "c2", "c3", "r1", "r2", "r3"}
    # entities are never expanded: the anchored entity "atlas" stays out
    assert engine.anchor_calls[0][1] == {"e1", "e2"}
    # hybrid's own anchored candidate first, then the window's rows in time order
    assert [chunk["id"] for chunk in result["chunks"]] == ["c1", "r1", "r3", "r2"]
    assert [entity["id"] for entity in result["entities"]] == ["e1", "e2"]
    assert [chunk["id"] for chunk in retriever.last_baseline["chunks"]] == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_window_expansion_spends_its_budget_on_the_tightest_timestamp_first(monkeypatch):
    """A one-day window also overlaps the bare-year node. 30 chunks hang off the year and
    one off the day; with a budget of 20 the day's chunk must still get in."""
    day = 86_400_000
    year_node = {"id": "ts_year", "timestamp_str": "1965", "time_at": 0, "time_until": 365 * day}
    day_node = {
        "id": "ts_day",
        "timestamp_str": "1965-03-18",
        "time_at": 76 * day,
        "time_until": 77 * day,
    }
    nodes = [(f"y{i}", {}) for i in range(30)] + [("voskhod", {})]
    edges = [(f"y{i}", "ts_year", "contains", {}) for i in range(30)] + [
        ("voskhod", "ts_day", "contains", {})
    ]
    rows = {
        "DocumentChunk_text": [{"id": f"y{i}", "text": "1965"} for i in range(30)]
        + [{"id": "voskhod", "text": "March 18, 1965"}]
    }
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(
            chunks=("voskhod",),
            timestamps=("ts_year", "ts_day"),
            chunk_timestamps={"voskhod": {"ts_day"}},
        ),
        interval=(_utc(1965, 3, 18), _utc(1965, 3, 19), None),
        candidates=_candidates(),
        timestamps=[year_node, day_node],  # the range lookup lists the year first
        neighborhood=(nodes, edges),
        vector_rows=rows,
    )
    result = await retriever.get_retrieved_objects(query="on 18 March 1965")
    assert "voskhod" in retriever.last_expansion
    assert len(retriever.last_expansion) == 20
    assert [chunk["id"] for chunk in result["chunks"]][0] == "voskhod"


@pytest.mark.asyncio
async def test_window_expansion_is_capped_at_candidate_top_k(monkeypatch):
    nodes = [(f"r{i}", {}) for i in range(50)]
    edges = [(f"r{i}", "ts_1950", "order_date", {}) for i in range(50)]
    rows = {"DltRow_text": [{"id": f"r{i}", "text": str(i)} for i in range(50)]}
    retriever, engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(chunks=tuple(f"r{i}" for i in range(50))),
        interval=WINDOW_1950,
        candidates=_candidates(),
        timestamps=[TS_1950],
        neighborhood=(nodes, edges),
        vector_rows=rows,
    )
    await retriever.get_retrieved_objects(query="in 1950")
    assert len(retriever.last_expansion) == retriever.chunks_top_k == 20


@pytest.mark.asyncio
async def test_window_expansion_respects_the_node_set_filter(monkeypatch):
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(chunks=("r3",)),
        interval=WINDOW_1950,
        candidates=_candidates(),
        timestamps=[TS_1950, TS_1950_06],
        neighborhood=_dated_graph(),
        vector_rows=_rows(),
    )
    retriever.node_name = ["other"]
    await retriever.get_retrieved_objects(query="in 1950")
    assert retriever.last_expansion == {"r3"}


@pytest.mark.asyncio
async def test_window_expansion_without_attached_nodes_keeps_the_fallback(monkeypatch):
    """A window whose timestamps have nothing retrievable attached is still a
    no_candidate_overlap fallback, not an error."""
    retriever, _engine, _fetch, _extract = _retriever(
        monkeypatch,
        anchors=_anchors(timestamps=()),
        interval=WINDOW_1950,
        candidates=_candidates(),
        timestamps=[TS_1950],
        neighborhood=([("tbl", {})], [("tbl", "ts_1950", "x", {})]),
        vector_rows={},
    )
    result = await retriever.get_retrieved_objects(query="in 1950")
    assert retriever.last_reason == "no_candidate_overlap"
    assert result == retriever.last_baseline


def test_attached_node_ids_orders_by_timestamp_and_skips_timestamps():
    from cognee.modules.retrieval.temporal_hybrid.expansion import attached_node_ids

    nodes, edges = _dated_graph()
    edges.append(
        ("ts_1950_06", "ts_1950", "follows", {})
    )  # timestamp to timestamp: never a candidate
    assert attached_node_ids([TS_1950, TS_1950_06], (nodes, edges)) == [
        "atlas",
        "c1",
        "r1",
        "r3",
        "tbl",
        "r2",
    ]


@pytest.mark.asyncio
async def test_retrieve_in_collections_keeps_id_order_and_first_collection_wins():
    from cognee.modules.retrieval.temporal_hybrid.expansion import retrieve_in_collections

    class Vector:
        async def retrieve(self, collection, ids):
            rows = {
                "A": [{"id": "x", "text": "from A"}, {"id": "y", "text": "y"}],
                "B": [{"id": "x", "text": "from B"}, {"id": "z", "text": "z"}],
            }[collection]
            return [row for row in rows if row["id"] in set(ids)]

    hits = await retrieve_in_collections(
        Vector(), ("A", "B"), ["z", "x", "missing", "y"], None, "OR"
    )
    assert [(hit["id"], hit["text"]) for hit in hits] == [("z", "z"), ("x", "from A"), ("y", "y")]
    assert await retrieve_in_collections(Vector(), ("A",), [], None, "OR") == []
