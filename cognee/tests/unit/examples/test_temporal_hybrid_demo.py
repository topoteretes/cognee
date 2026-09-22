import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

EXAMPLE_DIR = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "advanced_guides"
    / "temporal_awareness_example"
)


@pytest.fixture
def example_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    yield
    # monkeypatch restores sys.path but not sys.modules; drop the example
    # modules so their generic names (regex_chunker, ...) cannot shadow
    # same-named modules elsewhere in the session.
    for name, module in list(sys.modules.items()):
        if str(EXAMPLE_DIR) in str(getattr(module, "__file__", "") or ""):
            del sys.modules[name]


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


@pytest.mark.usefixtures("example_imports")
def test_load_queries_skips_blank_and_comment_lines(tmp_path):
    from temporal_hybrid_demo import load_queries

    path = tmp_path / "queries.txt"
    path.write_text(
        "# heading\n\nWhat happened on 27 April 1986?\n  \n# skip\nWho is MacDougall?\n",
        encoding="utf-8",
    )
    assert load_queries(path) == [
        "What happened on 27 April 1986?",
        "Who is MacDougall?",
    ]


@pytest.mark.usefixtures("example_imports")
def test_load_queries_rejects_empty_and_missing(tmp_path):
    from temporal_hybrid_demo import load_queries

    empty = tmp_path / "empty.txt"
    empty.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No queries"):
        load_queries(empty)
    with pytest.raises(RuntimeError, match="Cannot read demo file"):
        load_queries(tmp_path / "missing.txt")


@pytest.mark.usefixtures("example_imports")
def test_timestamp_bounds_four_precisions():
    from temporal_extraction_task import timestamp_bounds

    assert timestamp_bounds("  1950  ") == ("1950", _utc(1950, 1, 1), _utc(1951, 1, 1))
    assert timestamp_bounds("1950-03") == ("1950-03", _utc(1950, 3, 1), _utc(1950, 4, 1))
    assert timestamp_bounds("1950-12") == ("1950-12", _utc(1950, 12, 1), _utc(1951, 1, 1))
    assert timestamp_bounds("1950-03-15") == ("1950-03-15", _utc(1950, 3, 15), _utc(1950, 3, 16))
    assert timestamp_bounds("1950-03-15 12:30:45") == (
        "1950-03-15 12:30:45",
        _utc(1950, 3, 15, 12, 30, 45),
        _utc(1950, 3, 15, 12, 30, 46),
    )


@pytest.mark.usefixtures("example_imports")
def test_timestamp_bounds_leap_date():
    from temporal_extraction_task import timestamp_bounds

    assert timestamp_bounds("2020-02-29") == ("2020-02-29", _utc(2020, 2, 29), _utc(2020, 3, 1))
    with pytest.raises(ValueError):
        timestamp_bounds("2021-02-29")


@pytest.mark.usefixtures("example_imports")
def test_timestamp_bounds_invalid_and_overflow():
    from temporal_extraction_task import timestamp_bounds

    with pytest.raises(ValueError):
        timestamp_bounds("1940s")
    with pytest.raises(ValueError):
        timestamp_bounds("9999")


def _document():
    from uuid import uuid4

    from cognee.modules.data.processing.document_types import TextDocument

    return TextDocument(
        id=uuid4(),
        name="doc",
        raw_data_location="doc.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )


def _entity(name: str, type_name: str = "person"):
    from uuid import uuid4

    from cognee.modules.engine.models import Entity, EntityType

    return Entity(
        id=uuid4(),
        name=name,
        is_a=EntityType(name=type_name, description=type_name),
        description=name,
    )


def _chunk(document, contains, produced=None, text="text"):
    from uuid import uuid4

    from cognee.modules.chunking.models import DocumentChunk

    chunk = DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=1,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=list(contains),
    )
    if produced is not None:
        chunk._produced_edge_identities = list(produced)
    return chunk


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_imports")
async def test_regex_chunker_one_chunk_per_blank_line_section():
    from regex_chunker import RegexChunker

    text = "First section line one.\nStill first.\n\nSecond section.\n\n\n   \nThird section."

    async def get_text():
        yield text

    chunker = RegexChunker(_document(), get_text, max_chunk_size=512)
    chunks = [chunk async for chunk in chunker.read()]

    assert [chunk.text.strip() for chunk in chunks] == [
        "First section line one.\nStill first.",
        "Second section.",
        "Third section.",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.max_chunk_tokens == 512 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 3


@pytest.mark.usefixtures("example_imports")
def test_promote_timestamps_round_trip():
    from temporal_extraction_task import promote_timestamps

    from cognee.infrastructure.engine.models.Edge import Edge
    from cognee.modules.engine.models import Entity, Timestamp

    document = _document()
    person = _entity("Ada")
    stamp = _entity("1950", "Timestamp")
    other = _entity("Acme", "Organization")
    born = Edge(relationship_type="born_at")
    person.relations = [(born, stamp)]
    chunk_a = _chunk(
        document,
        [
            (Edge(relationship_type="contains"), person),
            (Edge(relationship_type="contains"), stamp),
        ],
    )
    chunk_b = _chunk(
        document,
        [
            (Edge(relationship_type="contains"), stamp),
            (Edge(relationship_type="contains"), other),
        ],
    )

    promote_timestamps([chunk_a, chunk_b])

    promoted = chunk_a.contains[1][1]
    assert isinstance(promoted, Timestamp)
    assert promoted.id == stamp.id
    assert promoted.timestamp_str == "1950"
    assert promoted is chunk_b.contains[0][1]
    assert person.relations[0][0] is born
    assert person.relations[0][1] is promoted
    assert isinstance(chunk_a.contains[0][1], Entity)
    assert chunk_a.contains[0][1] is person
    assert chunk_b.contains[1][1] is other


@pytest.mark.usefixtures("example_imports")
def test_promote_timestamps_skips_and_is_repeatable():
    pytest.importorskip("dateparser")
    from temporal_extraction_task import promote_timestamps

    from cognee.infrastructure.engine.models.Edge import Edge
    from cognee.modules.engine.models import Entity, Timestamp

    document = _document()
    valid = _entity("1950-03", "timestamp")
    unparseable = _entity("1940s", "Timestamp")
    outgoing = _entity("2000", "Timestamp")
    recorded = _entity("2001", "Timestamp")
    denormalized = _entity("23 March 1947", "Timestamp")
    person = _entity("Ada")
    outgoing.relations = [(Edge(relationship_type="related_to"), person)]
    chunk = _chunk(
        document,
        [
            (Edge(relationship_type="contains"), valid),
            (Edge(relationship_type="contains"), unparseable),
            (Edge(relationship_type="contains"), outgoing),
            (Edge(relationship_type="contains"), recorded),
            (Edge(relationship_type="contains"), denormalized),
        ],
        produced=[(str(recorded.id), str(person.id), "related_to")],
    )

    promote_timestamps([chunk])
    assert isinstance(chunk.contains[0][1], Timestamp)
    assert isinstance(chunk.contains[1][1], Entity)
    assert chunk.contains[1][1] is unparseable
    assert chunk.contains[2][1] is outgoing
    assert chunk.contains[3][1] is recorded
    assert isinstance(chunk.contains[4][1], Timestamp)
    assert chunk.contains[4][1].timestamp_str == "1947-03-23"
    assert chunk.contains[4][1].id == denormalized.id

    first = chunk.contains[0][1]
    promote_timestamps([chunk])
    assert chunk.contains[0][1] is first
    promote_timestamps([])
    empty = _chunk(document, [(Edge(relationship_type="contains"), person)])
    promote_timestamps([empty])
    assert empty.contains[0][1] is person


@pytest.mark.usefixtures("example_imports")
@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("23 March 1947", "1947-03-23"),
        ("March 1947", "1947-03"),
        ("April 27, 1791", "1791-04-27"),
        ("05:32 on 27 April 1986", "1986-04-27 05:32:00"),
        ("the 1950s", None),
        ("spring of 1943", None),
        ("four weeks later", None),
        ("that spring", None),
    ],
)
def test_normalize_absolute_date(name, normalized):
    pytest.importorskip("dateparser")
    from temporal_dateparser_hints import normalize_absolute_date

    assert normalize_absolute_date(name) == normalized


@pytest.mark.usefixtures("example_imports")
def test_hint_lines_rolls_base_and_marks_inferred():
    pytest.importorskip("dateparser")
    from temporal_dateparser_hints import hint_lines

    lines, base = hint_lines(
        "On 26 April 1986 the reactor exploded. The following night of 27 April, engineers worked.",
        None,
    )
    assert (base.year, base.month, base.day) == (1986, 4, 26)
    assert len(lines) == 1
    assert "1986-04-27" in lines[0]
    assert "inferred from context" in lines[0]


@pytest.mark.usefixtures("example_imports")
def test_hint_lines_without_stated_year_yields_nothing():
    pytest.importorskip("dateparser")
    from temporal_dateparser_hints import hint_lines

    lines, base = hint_lines("The following night of 27 April, engineers worked.", None)
    assert lines == []
    assert base is None


@pytest.mark.usefixtures("example_imports")
@pytest.mark.parametrize(
    ("span", "hinted"),
    [
        # date-anchored spans observed in the bundled data: keep
        ("at 4:00", True),
        ("05:32 UTC) on April 27", True),
        ("of 27 April", True),
        ("four weeks later", True),
        ("that spring", True),
        # scores, ordinals, durations, decades observed in the bundled data: reject
        ("1:2 at the", False),
        ("9-1", False),
        ("6th", False),
        ("of The year", False),
        ("3-2 and Second", False),
        ("1920s and", False),
        ("a four-year", False),
        ("ten years", False),
        ("two minutes", False),
        ("90 seconds and", False),
    ],
)
def test_date_reference_gate(span, hinted):
    from temporal_dateparser_hints import _looks_like_date_reference

    assert _looks_like_date_reference(span) is hinted


@pytest.mark.usefixtures("example_imports")
def test_hint_lines_ignores_scores_and_durations():
    pytest.importorskip("dateparser")
    from temporal_dateparser_hints import hint_lines

    lines, base = hint_lines(
        "On 26 April 1986 they won 9-1. The broadcast lasted two minutes.", None
    )
    assert lines == []
    assert (base.year, base.month, base.day) == (1986, 4, 26)


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_imports")
async def test_temporal_chunk_graphs_put_hints_in_prompt_only(monkeypatch):
    pytest.importorskip("dateparser")
    from temporal_dateparser_hints import calculate_temporal_chunk_graphs

    calls = []

    async def fake_extract(content, graph_model, custom_prompt=None, **kwargs):
        calls.append((content, custom_prompt, kwargs))
        return object()

    monkeypatch.setattr("temporal_dateparser_hints.extract_content_graph", fake_extract)
    document = _document()
    first = _chunk(document, [], text="On 26 April 1986 the reactor exploded.")
    second = _chunk(document, [], text="The following night of 27 April, engineers worked.")

    graphs = await calculate_temporal_chunk_graphs(
        [first, second], object, "PROMPT", calculate_chunk_graphs="self"
    )

    assert len(graphs) == len(calls) == 2
    (content_a, prompt_a, kwargs_a), (content_b, prompt_b, _) = calls
    assert content_a == first.text
    assert content_b == second.text
    assert prompt_a == "PROMPT"
    assert prompt_b.startswith("PROMPT\n\nTEMPORAL_NORMALIZATION_HINTS:")
    assert "1986-04-27" in prompt_b
    assert "calculate_chunk_graphs" not in kwargs_a

    with pytest.raises(ValueError, match="custom_prompt"):
        await calculate_temporal_chunk_graphs([first], object, None)


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
@pytest.mark.usefixtures("example_imports")
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
    from unittest.mock import AsyncMock

    from temporal_matching import extract_query_interval

    monkeypatch.setattr(
        "temporal_matching.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=_query_interval(start, end)),
    )
    got_start, got_end, got_reason = await extract_query_interval("query")
    assert (got_start, got_end, got_reason) == (expected_start, expected_end, reason)


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_imports")
async def test_extract_query_interval_propagates_llm_errors(monkeypatch):
    from unittest.mock import AsyncMock

    from temporal_matching import extract_query_interval

    monkeypatch.setattr(
        "temporal_matching.LLMGateway.acreate_structured_output",
        AsyncMock(side_effect=RuntimeError("provider")),
    )
    with pytest.raises(RuntimeError, match="provider"):
        await extract_query_interval("in 1950")


def _graph_node(node_id: str, type_name: str, **properties):
    return (node_id, {"type": type_name, **properties})


def _graph_edge(source: str, target: str, relationship: str):
    return (source, target, relationship, {})


@pytest.mark.usefixtures("example_imports")
def test_match_temporal_neighborhood_overlap_table():
    from temporal_matching import build_temporal_index, match_temporal_neighborhood

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


@pytest.mark.usefixtures("example_imports")
def test_filter_hybrid_relabels_timestamp_bullets():
    from temporal_matching import build_temporal_index, filter_hybrid, match_temporal_neighborhood

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
    from unittest.mock import AsyncMock

    from temporal_hybrid_retriever import TemporalHybridRetriever

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
    monkeypatch.setattr("temporal_hybrid_retriever.get_graph_engine", fake_graph_engine)
    monkeypatch.setattr("temporal_hybrid_retriever.get_unified_engine", fake_unified_engine)
    monkeypatch.setattr("temporal_hybrid_retriever.extract_query_interval", extract)
    monkeypatch.setattr(
        "temporal_hybrid_retriever.HybridRetriever.get_retrieved_objects", hybrid_fetch
    )
    retriever = TemporalHybridRetriever(candidate_top_k=20, top_k=1)
    return retriever, FakeGraphEngine, hybrid_fetch, extract


@pytest.mark.usefixtures("example_imports")
def test_temporal_retriever_rejects_bad_limits():
    from temporal_hybrid_retriever import TemporalHybridRetriever

    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=1, top_k=2)
    with pytest.raises(ValueError):
        TemporalHybridRetriever(candidate_top_k=5, top_k=0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_imports")
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
@pytest.mark.usefixtures("example_imports")
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
@pytest.mark.usefixtures("example_imports")
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
@pytest.mark.usefixtures("example_imports")
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
