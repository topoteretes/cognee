"""POST /api/v1/search and /recall forward includeGlobalContextIndex to the retriever.

The retrievers already read ``include_global_context_index`` from
``retriever_specific_config``; the HTTP bodies had no field for it, so the index
``improve(build_global_context_index=True)`` builds was unreachable over the API.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# The routers package __init__ re-exports the same-named factory, so import by dotted path.
search_router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")


def _client(router_module, router, prefix):
    app = FastAPI()
    app.include_router(router, prefix=prefix)
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return TestClient(app)


@pytest.fixture
def search_calls(monkeypatch):
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(search_router_module, "send_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.search"), "search", fake_search)
    client = _client(
        search_router_module, search_router_module.get_search_router(), "/api/v1/search"
    )
    return client, calls


@pytest.fixture
def recall_calls(monkeypatch):
    calls = []

    async def fake_recall(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.recall"), "recall", fake_recall)
    client = _client(
        recall_router_module, recall_router_module.get_recall_router(), "/api/v1/recall"
    )
    return client, calls


@pytest.mark.parametrize("field", ["includeGlobalContextIndex", "include_global_context_index"])
def test_search_forwards_the_flag(search_calls, field):
    client, calls = search_calls

    response = client.post("/api/v1/search", json={"query": "What is this about?", field: True})

    assert response.status_code == 200, response.text
    assert calls[0]["retriever_specific_config"] == {"include_global_context_index": True}


def test_search_without_the_flag_passes_no_config(search_calls):
    client, calls = search_calls

    response = client.post("/api/v1/search", json={"query": "What is this about?"})

    assert response.status_code == 200, response.text
    assert calls[0]["retriever_specific_config"] is None


def test_recall_forwards_the_flag(recall_calls):
    client, calls = recall_calls

    response = client.post(
        "/api/v1/recall", json={"query": "What is this about?", "includeGlobalContextIndex": True}
    )

    assert response.status_code == 200, response.text
    assert calls[0]["retriever_specific_config"] == {"include_global_context_index": True}


def test_recall_without_the_flag_passes_no_config(recall_calls):
    client, calls = recall_calls

    response = client.post("/api/v1/recall", json={"query": "What is this about?"})

    assert response.status_code == 200, response.text
    assert calls[0]["retriever_specific_config"] is None
