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

from cognee.modules.search.types import SearchType

recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")


@pytest.fixture
def client_and_calls(monkeypatch):
    calls = []

    async def fake_recall(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.recall"), "recall", fake_recall)

    app = FastAPI()
    app.include_router(recall_router_module.get_recall_router(), prefix="/api/v1/recall")
    app.dependency_overrides[recall_router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return TestClient(app), calls


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
