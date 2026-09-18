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

from cognee.modules.retrieval.exceptions.exceptions import NoDataError
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
    only_context_flags = []
    logged: list[str] = []
    script: dict[SearchType, object] = {}

    async def fake_authorized_search(*args, **kwargs):
        query_type = kwargs["query_type"]
        calls.append(query_type)
        only_context_flags.append(kwargs["only_context"])
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
    return SimpleNamespace(
        client=_build_client(),
        calls=calls,
        only_context_flags=only_context_flags,
        logged=logged,
        script=script,
    )


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


def test_cypher_text_is_answered_not_executed(retry_client):
    """Pasted Cypher reaches the default type, never the CYPHER retriever.

    The router cannot pick CYPHER, so a destructive statement arriving in a
    request body is treated as question text rather than run against the graph.
    """
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    response = retry_client.client.post(
        "/api/v1/recall", json={"query": "MATCH (n) DETACH DELETE n", "scope": "graph"}
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.HYBRID_COMPLETION]


def test_rejected_routed_type_falls_back_but_a_pinned_one_raises(retry_client):
    """A backend rejection of a guess is not the caller's mistake; of a pin, it is."""
    retry_client.script[SearchType.CODING_RULES] = UnsupportedSearchTypeError(
        "Coding rules search is disabled."
    )
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    routed = retry_client.client.post(
        "/api/v1/recall", json={"query": "what are our coding rules?", "scope": "graph"}
    )

    assert routed.status_code == 200, routed.text
    assert retry_client.calls == [SearchType.CODING_RULES, SearchType.HYBRID_COMPLETION]

    retry_client.calls.clear()
    with pytest.raises(UnsupportedSearchTypeError):
        retry_client.client.post(
            "/api/v1/recall",
            json={
                "query": "what are our coding rules?",
                "scope": "graph",
                "searchType": "CODING_RULES",
            },
        )
    assert retry_client.calls == [SearchType.CODING_RULES]


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


def _no_llm(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "llm_available", lambda _config: False
    )


def test_no_llm_default_is_hybrid_as_only_context(retry_client, monkeypatch):
    """With no LLM the router still routes; the default becomes HYBRID_COMPLETION run
    with only_context=True, so recall returns the assembled prompt instead of an
    answer. Keyed on LLM availability, not on the extractor that built the graph."""
    _no_llm(monkeypatch)
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    response = retry_client.client.post(
        "/api/v1/recall", json={"query": "where was Marie Curie born?", "scope": "graph"}
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.HYBRID_COMPLETION]
    assert retry_client.only_context_flags == [True]


def test_no_llm_still_routes_to_llm_free_types(retry_client, monkeypatch):
    """A rule whose target never calls an LLM runs as it is, and its empty-result
    fallback is the no-LLM default: HYBRID with only_context=True."""
    _no_llm(monkeypatch)
    retry_client.script[SearchType.CODING_RULES] = []
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    response = retry_client.client.post(
        "/api/v1/recall", json={"query": "what are our coding rules?", "scope": "graph"}
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.CODING_RULES, SearchType.HYBRID_COMPLETION]
    assert retry_client.only_context_flags == [False, True]


def test_no_llm_still_honours_a_pinned_type(retry_client, monkeypatch):
    """The constraints shape routing, not explicit choices: a pinned type runs as
    asked, with the caller's own only_context."""
    _no_llm(monkeypatch)
    retry_client.script[SearchType.RAG_COMPLETION] = _GRAPH_HIT

    response = retry_client.client.post(
        "/api/v1/recall",
        json={"query": "anything", "scope": "graph", "searchType": "RAG_COMPLETION"},
    )

    assert response.status_code == 200, response.text
    assert retry_client.calls == [SearchType.RAG_COMPLETION]
    assert retry_client.only_context_flags == [False]


def test_callers_only_context_is_kept_when_an_llm_is_available(retry_client):
    retry_client.script[SearchType.HYBRID_COMPLETION] = _GRAPH_HIT

    response = retry_client.client.post(
        "/api/v1/recall", json={"query": "anything", "scope": "graph", "onlyContext": True}
    )

    assert response.status_code == 200, response.text
    assert retry_client.only_context_flags == [True]


def test_an_empty_graph_surfaces_through_the_fallback(retry_client):
    """SDK-270 made an empty graph raise instead of returning []; the retry
    must surface that rather than swallow it into an empty 200.

    CODING_RULES returns [] on an empty graph, so the retry fires and lands on
    HYBRID_COMPLETION, which is one of the types that raises NoDataError. The
    error is not in the retry's except tuple, so it propagates — the caller is
    told the graph is empty instead of getting a silent miss.
    """
    retry_client.script[SearchType.HYBRID_COMPLETION] = NoDataError(
        "The knowledge graph is empty. Ingest data through Cognee before searching."
    )

    with pytest.raises(NoDataError):
        retry_client.client.post(
            "/api/v1/recall", json={"query": "what are our coding rules?", "scope": "graph"}
        )

    assert retry_client.calls == [SearchType.CODING_RULES, SearchType.HYBRID_COMPLETION]
    assert retry_client.logged == []


def test_a_pinned_type_never_reaches_the_empty_graph_error(retry_client):
    """The mirror of the above: pinning keeps the fallback out of the path.

    Same query, same empty graph, but CODING_RULES answers alone — so the caller
    gets a quiet empty result and never learns the graph is empty. That is the
    "a pinned type is never second-guessed" invariant costing information, not a
    bug, and it is worth having pinned so the asymmetry is a decision.
    """
    retry_client.script[SearchType.HYBRID_COMPLETION] = NoDataError("empty")

    response = retry_client.client.post(
        "/api/v1/recall",
        json={
            "query": "what are our coding rules?",
            "scope": "graph",
            "searchType": "CODING_RULES",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == []
    assert retry_client.calls == [SearchType.CODING_RULES]


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
