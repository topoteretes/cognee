"""The visualize routes check read permission once per request.

Each route authorized the dataset itself and then called an SDK function that
authorized it again (SDK-972), which is a handful of sequential permission
queries paid twice. The routers now hand the SDK function the Dataset they
authorized. Only the edges are faked here (permission lookup, graph read,
event collection, HTML and semantic rendering), so a second check anywhere in
the route or the SDK function counts as a second call.
"""

import sys
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_authenticated_user

router_module = import_module("cognee.api.v1.users.routers.get_visualize_router")
visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]

DATASET_ID = UUID("11111111-1111-4111-8111-111111111111")
EVENTS = [{"kind": "search", "qa_id": "q1"}]


@pytest.fixture
def client(monkeypatch):
    dataset = Dataset(id=DATASET_ID, name="some-dataset", owner_id=uuid4())
    authorize = AsyncMock(return_value=[dataset])
    fetch_graph = AsyncMock(return_value=([], []))
    collect_events = AsyncMock(return_value=EVENTS)
    render_html = AsyncMock(return_value="<html></html>")
    build_semantic = AsyncMock(return_value={"semantic_positions": None})

    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    # One mock behind both names, so a check in the route and another in the
    # SDK function show up as two awaits of it.
    monkeypatch.setattr(router_module, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(visualize_module, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(visualize_module, "fetch_dataset_graph_data", fetch_graph)
    monkeypatch.setattr(visualize_module, "collect_session_events", collect_events)
    monkeypatch.setattr(visualize_module, "cognee_network_visualization", render_html)
    monkeypatch.setattr(visualize_module, "build_semantic_payload", build_semantic)

    app = FastAPI()
    app.include_router(router_module.get_visualize_router(), prefix="/api/v1/visualize")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    with TestClient(app) as test_client:
        test_client.dataset = dataset
        test_client.authorize = authorize
        test_client.fetch_graph = fetch_graph
        test_client.collect_events = collect_events
        test_client.render_html = render_html
        yield test_client


@pytest.mark.parametrize("route", ["", "/json", "/semantic"])
def test_a_route_authorizes_the_dataset_once_and_reads_that_dataset(client, route):
    response = client.get(f"/api/v1/visualize{route}?dataset_id={DATASET_ID}")

    assert response.status_code == 200
    client.authorize.assert_awaited_once_with([DATASET_ID], "read", ANY)
    assert client.fetch_graph.await_args.args[0] is client.dataset


def test_the_json_response_carries_the_session_events_of_that_dataset(client):
    response = client.get(f"/api/v1/visualize/json?dataset_id={DATASET_ID}")

    assert response.json()["search_events"] == EVENTS
    assert client.collect_events.await_args.kwargs["dataset_id"] == DATASET_ID


def test_the_html_page_carries_the_session_events(client):
    client.get(f"/api/v1/visualize?dataset_id={DATASET_ID}")

    assert client.render_html.await_args.kwargs["search_events"] == EVENTS


def test_the_json_route_can_leave_the_session_events_out(client):
    response = client.get(
        f"/api/v1/visualize/json?dataset_id={DATASET_ID}&include_session_events=false"
    )

    assert response.status_code == 200
    assert response.json()["search_events"] == []
    client.collect_events.assert_not_awaited()
