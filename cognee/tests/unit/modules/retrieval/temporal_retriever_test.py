import asyncio
from types import SimpleNamespace
import pytest

from cognee.modules.retrieval.temporal_retriever import TemporalRetriever


# Test TemporalRetriever initialization defaults and overrides
def test_init_defaults_and_overrides():
    tr = TemporalRetriever()
    assert tr.top_k == 5
    assert tr.user_prompt_path == "graph_context_for_question.txt"
    assert tr.system_prompt_path == "answer_simple_question.txt"
    assert tr.time_extraction_prompt_path == "extract_query_time.txt"

    tr2 = TemporalRetriever(
        top_k=3,
        user_prompt_path="u.txt",
        system_prompt_path="s.txt",
        time_extraction_prompt_path="t.txt",
    )
    assert tr2.top_k == 3
    assert tr2.user_prompt_path == "u.txt"
    assert tr2.system_prompt_path == "s.txt"
    assert tr2.time_extraction_prompt_path == "t.txt"


# Test descriptions_to_string with basic and empty results
def test_descriptions_to_string_basic_and_empty():
    tr = TemporalRetriever()

    results = [
        {"description": "  First  "},
        {"nope": "no description"},
        {"description": "Second"},
        {"description": ""},
        {"description": "   Third line   "},
    ]

    s = tr.descriptions_to_string(results)
    assert s == "First\n#####################\nSecond\n#####################\nThird line"

    assert tr.descriptions_to_string([]) == ""


# Test filter_top_k_events sorts and limits correctly
@pytest.mark.asyncio
async def test_filter_top_k_events_sorts_and_limits():
    tr = TemporalRetriever(top_k=2)

    relevant_events = [
        {
            "events": [
                {"id": "e1", "description": "E1"},
                {"id": "e2", "description": "E2"},
                {"id": "e3", "description": "E3 - not in vector results"},
            ]
        }
    ]

    scored_results = [
        SimpleNamespace(payload={"id": "e2"}, score=0.10),
        SimpleNamespace(payload={"id": "e1"}, score=0.20),
    ]

    top = await tr.filter_top_k_events(relevant_events, scored_results)

    assert [e["id"] for e in top] == ["e2", "e1"]
    assert all("score" in e for e in top)
    assert top[0]["score"] == 0.10
    assert top[1]["score"] == 0.20


# Test filter_top_k_events handles unknown ids as infinite scores
@pytest.mark.asyncio
async def test_filter_top_k_events_includes_unknown_as_infinite_but_not_in_top_k():
    tr = TemporalRetriever(top_k=2)

    relevant_events = [
        {
            "events": [
                {"id": "known1", "description": "Known 1"},
                {"id": "unknown", "description": "Unknown"},
                {"id": "known2", "description": "Known 2"},
            ]
        }
    ]

    scored_results = [
        SimpleNamespace(payload={"id": "known2"}, score=0.05),
        SimpleNamespace(payload={"id": "known1"}, score=0.50),
    ]

    top = await tr.filter_top_k_events(relevant_events, scored_results)
    assert [e["id"] for e in top] == ["known2", "known1"]
    assert all(e["score"] != float("inf") for e in top)


# Test descriptions_to_string with unicode and newlines
def test_descriptions_to_string_unicode_and_newlines():
    tr = TemporalRetriever()
    results = [
        {"description": "Line A\nwith newline"},
        {"description": "This is a description"},
    ]
    s = tr.descriptions_to_string(results)
    assert "Line A\nwith newline" in s
    assert "This is a description" in s
    assert s.count("#####################") == 1


# Test filter_top_k_events when top_k is larger than available events
@pytest.mark.asyncio
async def test_filter_top_k_events_limits_when_top_k_exceeds_events():
    tr = TemporalRetriever(top_k=10)
    relevant_events = [{"events": [{"id": "a"}, {"id": "b"}]}]
    scored_results = [
        SimpleNamespace(payload={"id": "a"}, score=0.1),
        SimpleNamespace(payload={"id": "b"}, score=0.2),
    ]
    out = await tr.filter_top_k_events(relevant_events, scored_results)
    assert [e["id"] for e in out] == ["a", "b"]


# Test filter_top_k_events when scored_results is empty
@pytest.mark.asyncio
async def test_filter_top_k_events_handles_empty_scored_results():
    tr = TemporalRetriever(top_k=2)
    relevant_events = [{"events": [{"id": "x"}, {"id": "y"}]}]
    scored_results = []
    out = await tr.filter_top_k_events(relevant_events, scored_results)
    assert [e["id"] for e in out] == ["x", "y"]
    assert all(e["score"] == float("inf") for e in out)


# Test filter_top_k_events error handling for missing structure
@pytest.mark.asyncio
async def test_filter_top_k_events_error_handling():
    tr = TemporalRetriever(top_k=2)
    with pytest.raises((KeyError, TypeError)):
        await tr.filter_top_k_events([{}], [])


class _FakeRetriever(TemporalRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calls = []

    async def extract_time_from_query(self, query: str):
        if "both" in query:
            return "2024-01-01", "2024-12-31"
        if "from_only" in query:
            return "2024-01-01", None
        if "to_only" in query:
            return None, "2024-12-31"
        return None, None

    async def get_triplets(self, query: str):
        self._calls.append(("get_triplets", query))
        return [{"s": "a", "p": "b", "o": "c"}]

    async def resolve_edges_to_text(self, triplets):
        self._calls.append(("resolve_edges_to_text", len(triplets)))
        return "edges->text"

    async def _fake_graph_collect_ids(self, **kwargs):
        return ["e1", "e2"]

    async def _fake_graph_collect_events(self, ids):
        return [
            {
                "events": [
                    {"id": "e1", "description": "E1"},
                    {"id": "e2", "description": "E2"},
                    {"id": "e3", "description": "E3"},
                ]
            }
        ]

    async def _fake_vector_embed(self, texts):
        assert isinstance(texts, list) and texts
        return [[0.0, 1.0, 2.0]]

    async def _fake_vector_search(self, **kwargs):
        return [
            SimpleNamespace(payload={"id": "e2"}, score=0.05),
            SimpleNamespace(payload={"id": "e1"}, score=0.10),
        ]

    async def get_context(self, query: str):
        time_from, time_to = await self.extract_time_from_query(query)

        if not (time_from or time_to):
            triplets = await self.get_triplets(query)
            return await self.resolve_edges_to_text(triplets)

        ids = await self._fake_graph_collect_ids(time_from=time_from, time_to=time_to)
        relevant_events = await self._fake_graph_collect_events(ids)

        _ = await self._fake_vector_embed([query])
        vector_search_results = await self._fake_vector_search(
            collection_name="Event_name", query_vector=[0.0], limit=0
        )
        top_k_events = await self.filter_top_k_events(relevant_events, vector_search_results)
        return self.descriptions_to_string(top_k_events)


# Test get_context fallback to triplets when no time is extracted
@pytest.mark.asyncio
async def test_fake_get_context_falls_back_to_triplets_when_no_time():
    tr = _FakeRetriever(top_k=2)
    ctx = await tr.get_context("no_time")
    assert ctx == "edges->text"
    assert tr._calls[0][0] == "get_triplets"
    assert tr._calls[1][0] == "resolve_edges_to_text"


# Test get_context when time is extracted and vector ranking is applied
@pytest.mark.asyncio
async def test_fake_get_context_with_time_filters_and_vector_ranking():
    tr = _FakeRetriever(top_k=2)
    ctx = await tr.get_context("both time")
    assert ctx.startswith("E2")
    assert "#####################" in ctx
    assert "E1" in ctx and "E3" not in ctx
