"""POST /api/v1/recall auto-routes unless the caller pins a search type.

The DTO once defaulted ``search_type`` to HYBRID_COMPLETION, so the REST surface
was the only one that did not route by default. A body that omits the field must
now reach ``recall()`` with ``query_type=None`` (the router runs); an explicit
value must arrive as the enum member unchanged.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.search.exceptions import UnsupportedSearchTypeError
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(recall_router_module.get_recall_router(), prefix="/api/v1/recall")
    app.dependency_overrides[recall_router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return TestClient(app)


@pytest.fixture
def client_and_calls(monkeypatch):
    calls = []

    async def fake_recall(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.recall"), "recall", fake_recall)
    return _build_client(), calls


def test_omitted_search_type_auto_routes(client_and_calls):
    client, calls = client_and_calls

    response = client.post("/api/v1/recall", json={"query": "Summarize the report"})

    assert response.status_code == 200, response.text
    assert calls[0]["query_type"] is None


def test_explicit_null_search_type_auto_routes(client_and_calls):
    client, calls = client_and_calls

    response = client.post(
        "/api/v1/recall", json={"query": "Summarize the report", "searchType": None}
    )

    assert response.status_code == 200, response.text
    assert calls[0]["query_type"] is None


@pytest.mark.parametrize("field", ["search_type", "searchType"])
def test_explicit_search_type_is_pinned(client_and_calls, field):
    client, calls = client_and_calls

    response = client.post(
        "/api/v1/recall", json={"query": "Summarize the report", field: "CHUNKS"}
    )

    assert response.status_code == 200, response.text
    assert calls[0]["query_type"] is SearchType.CHUNKS


@pytest.fixture
def live_recall_client(monkeypatch):
    """A client wired to the real recall(), with only its two sources stubbed.

    ``client_and_calls`` replaces recall() wholesale, so it can assert which
    query_type arrives but never what the flip does downstream of it.
    """
    recall_mod = importlib.import_module("cognee.api.v1.recall.recall")

    async def fake_search_session(*args, **kwargs):
        return [
            recall_mod.ResponseQAEntry(
                time="2026-01-01T00:00:00+00:00",
                question="q",
                context="",
                answer="from the session",
                source="session",
            )
        ]

    async def fake_authorized_search(*args, **kwargs):
        return [
            SearchResultPayload(
                result_object="from the graph", search_type=SearchType.HYBRID_COMPLETION
            )
        ]

    async def fake_log_search_history(*args, **kwargs):
        return None

    monkeypatch.setattr(recall_mod, "_search_session", fake_search_session)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.search.methods.search"),
        "authorized_search",
        fake_authorized_search,
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.search.operations"),
        "log_search_history",
        fake_log_search_history,
    )
    return _build_client()


def test_omitted_search_type_makes_the_session_a_source(live_recall_client):
    """The undisclosed half of the flip: sessionId alone now answers from cache."""
    response = live_recall_client.post(
        "/api/v1/recall", json={"query": "what did we decide?", "sessionId": "s1"}
    )

    assert response.status_code == 200, response.text
    assert [entry["source"] for entry in response.json()] == ["session"]


def test_pinned_search_type_keeps_the_graph_as_the_only_source(live_recall_client):
    """Pinning a type restores the pre-PR behaviour for the same body."""
    response = live_recall_client.post(
        "/api/v1/recall",
        json={
            "query": "what did we decide?",
            "sessionId": "s1",
            "searchType": "HYBRID_COMPLETION",
        },
    )

    assert response.status_code == 200, response.text
    assert [entry["source"] for entry in response.json()] == ["graph"]


@pytest.fixture
def retry_client(monkeypatch):
    """A client on the real recall(), with the graph search scripted per type.

    ``calls`` records the search type of every ``authorized_search`` the request
    made, so a fallback shows up as a second entry, and ``logged`` captures the
    type search history recorded as having answered. ``script`` maps a search
    type to the results it returns, or to an exception it raises; anything
    unscripted returns no results.
    """
    calls: list[SearchType] = []
    logged: list[str] = []
    script: dict[SearchType, object] = {}

    async def fake_authorized_search(*args, **kwargs):
        query_type = kwargs["query_type"]
        calls.append(query_type)
        outcome = script.get(query_type, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def fake_log_search_history(query_text, search_type, *args, **kwargs):
        logged.append(search_type)

    monkeypatch.setattr(
        importlib.import_module("cognee.modules.search.methods.search"),
        "authorized_search",
        fake_authorized_search,
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.search.operations"),
        "log_search_history",
        fake_log_search_history,
    )
    return SimpleNamespace(client=_build_client(), calls=calls, logged=logged, script=script)


_GRAPH_HIT = [
    SearchResultPayload(result_object="from the graph", search_type=SearchType.HYBRID_COMPLETION)
]


def test_routed_type_with_no_results_falls_back_to_the_default(retry_client):
    """CODING_RULES on a dataset with no rules nodeset must not be the answer."""
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    response = retry_client.client.post(
        "/api/v1/recall", json={"query": "what are our coding rules?", "scope": "graph"}
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.CODING_RULES, SearchType.HYBRID_COMPLETION]
    assert retry_client.logged == ["HYBRID_COMPLETION"]


def test_pinned_type_with_no_results_is_not_retried(retry_client):
    """A type the caller chose is never second-guessed, empty or not."""
    response = retry_client.client.post(
        "/api/v1/recall",
        json={
            "query": "what are our coding rules?",
            "scope": "graph",
            "searchType": "CODING_RULES",
        },
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.CODING_RULES]


def test_routed_cypher_with_no_rows_is_not_retried(retry_client):
    """Zero rows is a correct Cypher answer, not an unavailable lane.

    Retrying would hand the LLM the Cypher text as a natural-language question.
    """
    response = retry_client.client.post(
        "/api/v1/recall", json={"query": "MATCH (n:Nonexistent) RETURN n", "scope": "graph"}
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.CYPHER]


def test_rejected_routed_type_falls_back_but_a_pinned_one_raises(retry_client):
    """ALLOW_CYPHER_QUERY=false is the deployment's choice, not the caller's mistake."""
    retry_client.script[SearchType.CYPHER] = UnsupportedSearchTypeError(
        "Cypher query search types are disabled."
    )
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    routed = retry_client.client.post(
        "/api/v1/recall", json={"query": "MATCH (n) RETURN n", "scope": "graph"}
    )

    assert routed.status_code == 200, routed.text
    assert retry_client.calls == [SearchType.CYPHER, SearchType.HYBRID_COMPLETION]

    retry_client.calls.clear()
    with pytest.raises(UnsupportedSearchTypeError):
        retry_client.client.post(
            "/api/v1/recall",
            json={"query": "MATCH (n) RETURN n", "scope": "graph", "searchType": "CYPHER"},
        )
    assert retry_client.calls == [SearchType.CYPHER]


def test_a_failure_of_the_default_type_is_not_swallowed(retry_client):
    """The router labels its own fallback "default", which once read as a guess.

    That made an unpinned request swallow the rejection and answer 200 with no
    results, while the identical pinned request raised.
    """
    retry_client.script[SearchType.HYBRID_COMPLETION] = UnsupportedSearchTypeError(
        "skills/tools are supported only with SearchType.AGENTIC_COMPLETION"
    )

    with pytest.raises(UnsupportedSearchTypeError):
        retry_client.client.post(
            "/api/v1/recall", json={"query": "what did we decide?", "scope": "graph"}
        )

    assert retry_client.calls == [SearchType.HYBRID_COMPLETION]


def test_session_first_scope_short_circuits_without_omitting_the_type(live_recall_client):
    """The short-circuit is reachable explicitly, so query_type need not carry it."""
    response = live_recall_client.post(
        "/api/v1/recall",
        json={
            "query": "what did we decide?",
            "sessionId": "s1",
            "scope": "session_first",
            "searchType": "HYBRID_COMPLETION",
        },
    )

    assert response.status_code == 200, response.text
    assert [entry["source"] for entry in response.json()] == ["session"]


def test_plain_session_scope_lets_both_sources_contribute(live_recall_client):
    """Without session_first, an explicit scope never short-circuits."""
    response = live_recall_client.post(
        "/api/v1/recall",
        json={
            "query": "what did we decide?",
            "sessionId": "s1",
            "scope": ["session", "graph"],
            "searchType": "HYBRID_COMPLETION",
        },
    )

    assert response.status_code == 200, response.text
    assert [entry["source"] for entry in response.json()] == ["session", "graph"]
