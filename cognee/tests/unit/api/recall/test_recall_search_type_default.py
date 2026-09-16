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
