"""Guards that `query` is required on POST /api/v1/recall and /api/v1/search.

Both DTOs once carried `query: str = Field(default="What is in the document?")`,
an OpenAPI example wired as a functional default. A body omitting `query`
validated, and the placeholder sentence was answered as though the caller had
asked it, across every dataset the caller could read. A regression that
reintroduces a default for `query` fails here.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

# The routers package __init__ re-exports the same-named factory function, so
# the module must be imported by its dotted path.
recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")
search_router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")


def _client(router_module, factory_name, prefix):
    app = FastAPI()
    app.include_router(getattr(router_module, factory_name)(), prefix=prefix)
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return TestClient(app)


def _spy(monkeypatch, package_path, attribute):
    """Replace the retrieval callable with a spy recording whether it ran."""
    calls = []

    async def fake(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(importlib.import_module(package_path), attribute, fake)
    return calls


def test_recall_rejects_body_without_query(monkeypatch):
    calls = _spy(monkeypatch, "cognee.api.v1.recall", "recall")
    client = _client(recall_router_module, "get_recall_router", "/api/v1/recall")

    response = client.post("/api/v1/recall", json={})

    assert response.status_code == 422, response.text
    assert calls == [], "retrieval ran for a request that omitted query"


def test_search_rejects_body_without_query(monkeypatch):
    calls = _spy(monkeypatch, "cognee.api.v1.search", "search")
    client = _client(search_router_module, "get_search_router", "/api/v1/search")

    response = client.post("/api/v1/search", json={})

    assert response.status_code == 422, response.text
    assert calls == [], "retrieval ran for a request that omitted query"


def test_recall_still_accepts_an_explicit_query(monkeypatch):
    calls = _spy(monkeypatch, "cognee.api.v1.recall", "recall")
    client = _client(recall_router_module, "get_recall_router", "/api/v1/recall")

    response = client.post("/api/v1/recall", json={"query": "who wrote it?"})

    assert response.status_code == 200, response.text
    assert calls and calls[0]["query_text"] == "who wrote it?"


def test_search_still_accepts_an_explicit_query(monkeypatch):
    calls = _spy(monkeypatch, "cognee.api.v1.search", "search")
    client = _client(search_router_module, "get_search_router", "/api/v1/search")

    response = client.post("/api/v1/search", json={"query": "who wrote it?"})

    assert response.status_code == 200, response.text
    assert calls and calls[0]["query_text"] == "who wrote it?"
