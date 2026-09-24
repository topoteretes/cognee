"""GET /visualize/json, streamed.

Streaming is opt-in, so every existing caller still gets the JSON it got
before. A streamed request that fails before its first chunk fails the way the
JSON one would, with a status code. And the node cap depends on the
transport: the JSON response holds the whole graph, the stream one chunk.
"""

import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.exceptions import CogneeApiError
from cognee.modules.users.methods import get_authenticated_user

router_module = import_module("cognee.api.v1.users.routers.get_visualize_router")
visualize_pkg = import_module("cognee.api.v1.visualize")

DATASET_ID = "11111111-1111-1111-1111-111111111111"
JSON_PAYLOAD = {"nodes": [{"id": "a"}], "links": [], "color_maps": {}}


def _events(*, fail_with=None):
    calls = []

    async def stream_dataset_graph(dataset, **kwargs):
        calls.append(kwargs)
        if fail_with is not None:
            raise fail_with
        yield "meta", {"seeds": ["a"], "seed_source": "explicit"}
        yield "chunk", {"index": 0, "nodes": [{"id": "a"}], "links": []}
        yield "summary", {"nodes": {"a": {"importance": 0.0, "label_priority": False}}}
        yield "done", {"nodes": 1, "links": 0, "chunks": 1}

    return stream_dataset_graph, calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(side_effect=lambda ids, _permission, _user: [SimpleNamespace(id=ids[0])]),
    )
    json_path = AsyncMock(return_value=JSON_PAYLOAD)
    monkeypatch.setattr(visualize_pkg, "visualize_graph_json", json_path)
    stream, calls = _events()
    monkeypatch.setattr(visualize_pkg, "stream_dataset_graph", stream)

    app = FastAPI()
    app.include_router(router_module.get_visualize_router(), prefix="/api/v1/visualize")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    with TestClient(app) as test_client:
        test_client.json_path = json_path
        test_client.stream_calls = calls
        yield test_client


def _get(client, query="", headers=None):
    return client.get(f"/api/v1/visualize/json?dataset_id={DATASET_ID}{query}", headers=headers)


def _event_names(response):
    return [line[7:] for line in response.text.splitlines() if line.startswith("event: ")]


def test_a_default_request_still_gets_the_json_payload(client):
    response = _get(client)

    assert response.status_code == 200
    assert response.json() == JSON_PAYLOAD
    assert client.stream_calls == []


@pytest.mark.parametrize(
    ("query", "headers", "streams"),
    [
        ("", {"Accept": "text/event-stream"}, True),
        ("&stream=true", None, True),
        ("&stream=false", {"Accept": "text/event-stream"}, False),
        # fetch and httpx send this; a tie must stay on JSON.
        ("", {"Accept": "*/*"}, False),
        ("", {"Accept": "application/json, text/event-stream"}, False),
    ],
)
def test_streaming_is_negotiated(client, query, headers, streams):
    response = _get(client, query, headers)

    assert response.status_code == 200
    assert ("text/event-stream" in response.headers["content-type"]) is streams
    assert bool(client.stream_calls) is streams


def test_a_streamed_request_gets_the_events_in_order(client):
    response = _get(client, "&stream=true&max_nodes=20000&neighborhood_depth=3")

    assert _event_names(response) == ["meta", "chunk", "summary", "done"]
    assert response.headers["x-accel-buffering"] == "no"
    assert "connection" not in response.headers
    assert client.stream_calls[0]["max_nodes"] == 20000
    assert client.stream_calls[0]["neighborhood_depth"] == 3


def test_more_than_the_json_cap_is_rejected_exactly_as_before_streaming_existed(client):
    """Before the stream, `le=5000` on the Query rejected this. A JSON caller
    must still get that same response, whatever the app's validation handler."""
    from fastapi import Query

    before = FastAPI()

    @before.get("/json")
    async def old_route(max_nodes: int = Query(500, ge=1, le=5000)):
        return {}

    expected = TestClient(before).get("/json?max_nodes=5001")

    response = _get(client, "&max_nodes=5001")

    assert (response.status_code, response.json()) == (expected.status_code, expected.json())
    client.json_path.assert_not_awaited()


def test_the_json_cap_itself_is_still_json(client):
    assert _get(client, "&max_nodes=5000").status_code == 200
    client.json_path.assert_awaited_once()


def test_the_stream_cap_is_enforced(client):
    assert _get(client, "&stream=true&max_nodes=20001").status_code == 422
    assert client.stream_calls == []


def test_the_whole_graph_cannot_be_streamed(client):
    response = _get(client, "&stream=true&full=true")

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "full"]
    assert client.stream_calls == []


def test_the_stream_cap_and_the_json_cap_reject_alike(client):
    """Both bounds on max_nodes answer with the framework's validation shape."""
    over_json = _get(client, "&max_nodes=5001").json()
    over_stream = _get(client, "&stream=true&max_nodes=20001").json()

    assert (
        over_json["detail"][0]["loc"] == over_stream["detail"][0]["loc"] == ["query", "max_nodes"]
    )


def test_a_full_stream_pool_is_a_503_before_anything_is_read(client, monkeypatch):
    import asyncio

    from cognee.modules.visualization import graph_stream
    from cognee.modules.visualization.exceptions import GraphStreamCapacityError

    monkeypatch.setattr(graph_stream, "_permits", asyncio.Semaphore(0))

    # No global handler in this app: the cognee server's CogneeApiError handler
    # turns this into the 503, as it does for the JSON path's own errors.
    with pytest.raises(GraphStreamCapacityError) as refused:
        _get(client, "&stream=true")
    assert refused.value.status_code == 503
    assert client.stream_calls == []


def test_a_failure_before_the_first_chunk_keeps_its_status_code(client, monkeypatch):
    stream, _ = _events(fail_with=RuntimeError("password=hunter2"))
    monkeypatch.setattr(visualize_pkg, "stream_dataset_graph", stream)

    response = _get(client, "&stream=true")

    assert response.status_code == 409
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert "hunter2" not in response.text


def test_a_cognee_error_before_the_first_chunk_is_left_to_the_global_handler(client, monkeypatch):
    """CogneeApiError subclasses carry their own status and the JSON path re-raises them."""

    class _Denied(CogneeApiError):
        def __init__(self):
            super().__init__(message="No access to dataset", status_code=403)

    stream, _ = _events(fail_with=_Denied())
    monkeypatch.setattr(visualize_pkg, "stream_dataset_graph", stream)

    with pytest.raises(_Denied):
        _get(client, "&stream=true")


def test_the_streamed_chunk_is_valid_json(client):
    response = _get(client, "&stream=true")

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert json.loads(data_lines[1]) == {"index": 0, "nodes": [{"id": "a"}], "links": []}
