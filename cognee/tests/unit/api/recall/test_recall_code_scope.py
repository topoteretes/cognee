"""The explicit "code" recall scope runs the deterministic code-graph lane.

The scope is opt-in only — never implied by "auto" or "all" — and delegates to
``authorized_search`` with ``SearchType.CODE`` pinned, so all tests stub that
one seam: no databases, graph engines, or LLMs are involved.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.api.v1.recall.recall import recall
from cognee.exceptions import CogneeValidationError
from cognee.memory.entries import normalize_scope
from cognee.modules.retrieval.code_retriever import CodeSeedNotFoundError
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

# cognee.modules.search.methods re-exports `search` the function, which shadows
# the module of the same name for dotted-path patching.
search_mod = importlib.import_module("cognee.modules.search.methods.search")
recall_mod = importlib.import_module("cognee.api.v1.recall.recall")
history_mod = importlib.import_module("cognee.modules.search.operations.log_search_history")


def code_payload(completion, dataset_id=None):
    return SearchResultPayload(
        dataset_id=dataset_id or uuid4(),
        completion=completion,
        search_type=SearchType.CODE,
    )


@pytest.fixture
def captured_search(monkeypatch):
    """Stub authorized_search, capturing one kwargs dict per call."""
    calls = []
    results = {"value": []}

    async def fake_authorized_search(**kwargs):
        calls.append(kwargs)
        value = results["value"]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(search_mod, "authorized_search", fake_authorized_search)
    return {"calls": calls, "results": results}


@pytest.fixture
def graph_lane_stubs(monkeypatch):
    """Silence the graph lane's side channels (history log, session context)."""

    async def fake_log_query(*_args, **_kwargs):
        return SimpleNamespace(id=uuid4())

    async def fake_log_result(*_args, **_kwargs):
        return None

    async def fake_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(history_mod, "log_query", fake_log_query)
    monkeypatch.setattr(history_mod, "log_result", fake_log_result)
    monkeypatch.setattr(
        recall_mod, "set_session_user_context_variable", fake_set_session_user_context_variable
    )


@pytest.mark.asyncio
async def test_code_scope_runs_code_lane_and_tags_source(captured_search):
    user = SimpleNamespace(id=uuid4())
    captured_search["results"]["value"] = [
        code_payload({"nodes": [{"name": "UserService", "file": "services/user.py"}]})
    ]

    results = await recall(
        "UserService",
        scope=["code"],
        dataset_ids=[uuid4()],
        top_k=7,
        user=user,
    )

    assert len(captured_search["calls"]) == 1
    call = captured_search["calls"][0]
    assert call["query_type"] is SearchType.CODE
    assert call["query_text"] == "UserService"
    assert call["top_k"] == 7
    assert call["retriever_specific_config"] is None

    assert results
    assert all(entry.source == "code" for entry in results)


@pytest.mark.asyncio
async def test_code_query_is_forwarded_as_retriever_config(captured_search):
    code_query = {"operation": "impact_analysis", "seeds": ["UserService"], "max_depth": 2}
    captured_search["results"]["value"] = [code_payload({"impact": []})]

    await recall(
        "unused seed text",
        scope=["code"],
        dataset_ids=[uuid4()],
        code_query=code_query,
        user=SimpleNamespace(id=uuid4()),
    )

    assert captured_search["calls"][0]["retriever_specific_config"] == code_query


@pytest.mark.asyncio
async def test_code_query_without_code_scope_is_rejected(captured_search):
    with pytest.raises(CogneeValidationError, match="code_query requires the 'code' scope"):
        await recall(
            "UserService",
            scope=["graph"],
            code_query={"operation": "explore"},
            user=SimpleNamespace(id=uuid4()),
        )
    # Auto scope never implies "code" either.
    with pytest.raises(CogneeValidationError, match="code_query requires the 'code' scope"):
        await recall(
            "UserService",
            code_query={"operation": "explore"},
            user=SimpleNamespace(id=uuid4()),
        )
    assert not captured_search["calls"]


@pytest.mark.asyncio
async def test_seed_not_found_contributes_nothing(captured_search):
    captured_search["results"]["value"] = CodeSeedNotFoundError("seed 'Xyz' not found")

    results = await recall(
        "Xyz",
        scope=["code"],
        dataset_ids=[uuid4()],
        user=SimpleNamespace(id=uuid4()),
    )

    assert results == []


@pytest.mark.asyncio
async def test_softened_per_dataset_seed_misses_are_dropped(captured_search):
    captured_search["results"]["value"] = [
        code_payload({"seed_not_found": True, "error": "seed 'Xyz' not found"}),
        code_payload({"nodes": [{"name": "Xyz"}]}),
    ]

    results = await recall(
        "Xyz",
        scope=["code"],
        dataset_ids=[uuid4(), uuid4()],
        user=SimpleNamespace(id=uuid4()),
    )

    assert results
    assert all(entry.source == "code" for entry in results)
    assert all(not entry.raw.get("seed_not_found") for entry in results)


@pytest.mark.asyncio
async def test_graph_and_code_lanes_both_contribute(captured_search, graph_lane_stubs, monkeypatch):
    graph_result = [
        SearchResultPayload(
            dataset_id=uuid4(),
            completion="the graph answer",
            code_evidence={
                "sources": [{"name": "UserService"}],
                "operations": [{"operation": "ask", "seeds": [{"name": "UserService"}]}],
            },
            search_type=SearchType.GRAPH_COMPLETION,
        )
    ]

    async def routed(**kwargs):
        captured_search["calls"].append(kwargs)
        return graph_result

    monkeypatch.setattr(search_mod, "authorized_search", routed)

    results = await recall(
        "what calls UserService?",
        query_type=SearchType.GRAPH_COMPLETION,
        auto_route=False,
        scope=["graph", "code"],
        dataset_ids=[uuid4()],
        user=SimpleNamespace(id=uuid4()),
    )

    lanes = {call["query_type"] for call in captured_search["calls"]}
    assert lanes == {SearchType.GRAPH_COMPLETION}
    config = captured_search["calls"][0]["retriever_specific_config"]
    assert config == {"code_context": None, "code_plan_cache": {}}
    assert {entry.source for entry in results} == {"graph", "code"}
    assert results[0].metadata["code_evidence"]["sources"][0]["name"] == "UserService"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_type",
    [
        SearchType.TEMPORAL,
        SearchType.FEELING_LUCKY,
        SearchType.BROAD,
        SearchType.AGENTIC_COMPLETION,
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
    ],
)
async def test_an_answer_type_that_cannot_carry_code_evidence_is_refused(
    captured_search, graph_lane_stubs, search_type
):
    with pytest.raises(CogneeValidationError, match="Combined graph/code answers"):
        await recall(
            "what calls UserService?",
            query_type=search_type,
            auto_route=False,
            scope=["graph", "code"],
            dataset_ids=[uuid4()],
            user=SimpleNamespace(id=uuid4()),
        )
    assert not captured_search["calls"]


@pytest.mark.asyncio
async def test_a_routed_temporal_type_answers_with_code_instead_of_failing(
    captured_search, graph_lane_stubs, monkeypatch
):
    captured_search["results"]["value"] = []
    await recall(
        "What did the sync code change in 2019?",
        scope=["graph", "code"],
        dataset_ids=[uuid4()],
        user=SimpleNamespace(id=uuid4()),
    )
    types = {call["query_type"] for call in captured_search["calls"]}
    assert SearchType.TEMPORAL not in types
    assert types & {SearchType.HYBRID_COMPLETION, SearchType.GRAPH_COMPLETION}


@pytest.mark.asyncio
async def test_a_retrieval_only_type_still_returns_graph_and_code_results(
    captured_search, graph_lane_stubs
):
    captured_search["results"]["value"] = [
        SearchResultPayload(dataset_id=uuid4(), completion=["chunk"], search_type=SearchType.CHUNKS)
    ]
    await recall(
        "UserService",
        query_type=SearchType.CHUNKS,
        auto_route=False,
        scope=["graph", "code"],
        dataset_ids=[uuid4()],
        user=SimpleNamespace(id=uuid4()),
    )
    assert {call["query_type"] for call in captured_search["calls"]} == {
        SearchType.CHUNKS,
        SearchType.CODE,
    }
    assert all(
        "code_context" not in (call["retriever_specific_config"] or {})
        for call in captured_search["calls"]
    )


def test_code_is_a_valid_scope_but_never_implied():
    assert normalize_scope(["graph", "code"]) == ["graph", "code"]
    assert normalize_scope("code") == ["code"]
    # "all" expands to every passive source; "code" (like "tools") stays opt-in.
    assert "code" not in normalize_scope("all")
    assert normalize_scope(None) == ["auto"]
