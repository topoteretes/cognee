"""POST /api/v1/recall and /api/v1/search accept search_type=TEMPORAL (SDK-829).

The DTOs type the field as the SearchType enum, so this guards the enum member
staying reachable over REST rather than any parsing of its own.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.search.types import SearchType

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
    calls = []

    async def fake(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(importlib.import_module(package_path), attribute, fake)
    return calls


def test_recall_accepts_temporal(monkeypatch):
    calls = _spy(monkeypatch, "cognee.api.v1.recall", "recall")
    client = _client(recall_router_module, "get_recall_router", "/api/v1/recall")

    response = client.post(
        "/api/v1/recall", json={"query": "what happened in 1898?", "searchType": "TEMPORAL"}
    )

    assert response.status_code == 200, response.text
    assert calls and calls[0]["query_type"] is SearchType.TEMPORAL


def test_search_accepts_temporal(monkeypatch):
    calls = _spy(monkeypatch, "cognee.api.v1.search", "search")
    client = _client(search_router_module, "get_search_router", "/api/v1/search")

    response = client.post(
        "/api/v1/search", json={"query": "what happened in 1898?", "searchType": "TEMPORAL"}
    )

    assert response.status_code == 200, response.text
    assert calls and calls[0]["query_type"] is SearchType.TEMPORAL
