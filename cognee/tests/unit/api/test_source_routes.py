import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

routes = importlib.import_module("cognee.api.v1.datasets.routers.source_routes")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.get_source_routes(), prefix="/api/v1/datasets")
    app.dependency_overrides[routes.get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    return TestClient(app)


def test_catalog_pagination(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "source_catalog",
        AsyncMock(return_value={"items": [{"id": str(i)} for i in range(5)], "complete": True}),
    )
    result = client.get("/api/v1/datasets/source-catalog?offset=2&limit=2")
    assert result.status_code == 200
    assert result.json()["items"] == [{"id": "2"}, {"id": "3"}]
    assert result.json()["next_offset"] == 4


def test_route_denial_is_not_retried_as_owner(client, monkeypatch):
    method = AsyncMock(side_effect=PermissionError())
    monkeypatch.setattr(routes, "route_sources", method)
    result = client.post("/api/v1/datasets/source-route", json={"query": "anything"})
    assert result.status_code == 403
    method.assert_awaited_once()


def test_route_failure_does_not_expose_provider_details(client, monkeypatch):
    monkeypatch.setattr(
        routes, "route_sources", AsyncMock(side_effect=RuntimeError("secret token"))
    )
    result = client.post("/api/v1/datasets/source-route", json={"query": "anything"})
    assert result.status_code == 409
    assert "secret token" not in result.text


def test_route_rejects_unbounded_budgets(client):
    assert (
        client.post(
            "/api/v1/datasets/source-route",
            json={"query": "anything", "max_catalog_entries": 999999},
        ).status_code
        == 422
    )


def test_source_document_uses_explicit_dataset_and_identity(client, monkeypatch):
    dataset, document = uuid4(), uuid4()
    method = AsyncMock(return_value={"id": str(document)})
    monkeypatch.setattr(routes, "source_document", method)
    result = client.get(f"/api/v1/datasets/source-document/{dataset}/{document}")
    assert result.status_code == 200
    assert method.await_args.args[1:] == (dataset, document)


@pytest.mark.parametrize("kind", ["search", "recall"])
def test_http_node_set_operator_is_validated(kind):
    module = importlib.import_module(f"cognee.api.v1.{kind}.routers.get_{kind}_router")
    dto = getattr(module, f"{kind.title()}PayloadDTO")
    assert dto(query="q", node_name_filter_operator="AND").node_name_filter_operator == "AND"
    with pytest.raises(ValueError):
        dto(query="q", node_name_filter_operator="SQL")


@pytest.mark.parametrize("kind", ["search", "recall"])
@pytest.mark.parametrize("operator", ["AND", "OR"])
def test_http_forwards_node_set_operator_to_native_sdk(kind, operator, monkeypatch):
    module = importlib.import_module(f"cognee.api.v1.{kind}.routers.get_{kind}_router")
    native = importlib.import_module(f"cognee.api.v1.{kind}")
    method = AsyncMock(return_value=[])
    monkeypatch.setattr(native, kind, method)
    monkeypatch.setattr(module, "send_telemetry", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(getattr(module, f"get_{kind}_router")(), prefix=f"/api/v1/{kind}")
    user = SimpleNamespace(id=uuid4())
    app.dependency_overrides[module.get_authenticated_user] = lambda: user
    result = TestClient(app).post(
        f"/api/v1/{kind}",
        json={
            "query": "q",
            "search_type": "CHUNKS",
            "node_name": ["one", "two"],
            "node_name_filter_operator": operator,
        },
    )
    assert result.status_code == 200, result.text
    assert method.await_args.kwargs["node_name_filter_operator"] == operator
    assert method.await_args.kwargs["node_name"] == ["one", "two"]
    assert method.await_args.kwargs["user"] is user
