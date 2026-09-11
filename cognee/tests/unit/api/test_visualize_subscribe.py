"""WS /visualize/subscribe/{dataset_id}: the push replacement for polling.

Driven through the real router over a TestClient, with the polls themselves
patched (get_live_events has its own tests, and the pipeline-run read is one
existing query) and the loop's tick shortened. What this pins is the wire
protocol a client is written against — the frame shapes, that a delta is only
sent when it is non-empty, that a run already complete at connect time is not
announced as growth, that the cursor advances so nothing replays — and the
close codes that tell a client whether to retry.
"""

from datetime import datetime
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_websocket_user
from cognee.modules.pipelines.models import PipelineRunStatus

router_module = import_module("cognee.api.v1.users.routers.get_visualize_router")
live_updates = import_module("cognee.api.v1.visualize.live_updates")

DATASET_ID = "11111111-1111-1111-1111-111111111111"


def _no_events(_dataset_id, since=None, user=None):
    return {"events": [], "cursor": since.isoformat() if since else None}


@pytest.fixture
def app(monkeypatch):
    """The real router, with the loop sped up and its polls neutralized."""
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(side_effect=lambda ids, _permission, _user: [SimpleNamespace(id=ids[0])]),
    )
    monkeypatch.setattr(
        live_updates,
        "get_authorized_existing_datasets",
        AsyncMock(side_effect=lambda ids, _permission, _user: [SimpleNamespace(id=ids[0])]),
    )
    monkeypatch.setattr(live_updates, "TICK_SECONDS", 0.005)
    monkeypatch.setattr(live_updates, "get_live_events", AsyncMock(side_effect=_no_events))
    monkeypatch.setattr(live_updates, "get_pipeline_run_by_dataset", AsyncMock(return_value=None))
    # Every test reuses the same DATASET_ID: without this, one test's cached
    # latest-run-id would leak into the next via the shared module-level
    # cache that collapses duplicate polling across concurrent viewers.
    live_updates._reset_latest_run_cache()

    application = FastAPI()
    application.include_router(router_module.get_visualize_router(), prefix="/api/v1/visualize")
    application.dependency_overrides[get_authenticated_websocket_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return application


def _connect(app, query: str = ""):
    return TestClient(app).websocket_connect(f"/api/v1/visualize/subscribe/{DATASET_ID}{query}")


def _completed_run(pipeline_run_id):
    return SimpleNamespace(
        pipeline_run_id=pipeline_run_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
    )


def test_ready_is_the_first_frame_and_echoes_the_starting_cursor(app):
    with _connect(app) as connection:
        assert connection.receive_json() == {
            "kind": "ready",
            "dataset_id": DATASET_ID,
            "cursor": None,
        }

    with _connect(app, "?since=2026-08-03T09:00:05") as connection:
        assert connection.receive_json() == {
            "kind": "ready",
            "dataset_id": DATASET_ID,
            "cursor": "2026-08-03T09:00:05",
        }


def test_a_reconnect_cursor_is_passed_through_to_the_events_query(app, monkeypatch):
    seen = []

    async def record(dataset_id, since=None, user=None):
        seen.append(since)
        return {"events": [], "cursor": None}

    monkeypatch.setattr(live_updates, "get_live_events", record)

    with _connect(app, "?since=2026-08-03T09:00:05") as connection:
        connection.receive_json()
        # Wait for a frame that can only follow several poll cycles.
        while connection.receive_json()["kind"] != "heartbeat":
            pass

    assert seen and seen[0] == datetime(2026, 8, 3, 9, 0, 5)


def test_events_are_pushed_only_when_the_delta_is_non_empty(app, monkeypatch):
    events = [{"kind": "search", "time": "2026-08-03T09:00:10.000000", "qa_id": "a"}]
    deliveries = iter([{"events": events, "cursor": "2026-08-03T09:00:10.000000"}])

    async def once(_dataset_id, since=None, user=None):
        try:
            return next(deliveries)
        except StopIteration:
            return {"events": [], "cursor": since.isoformat() if since else None}

    monkeypatch.setattr(live_updates, "get_live_events", once)

    with _connect(app) as connection:
        connection.receive_json()
        frames = [connection.receive_json() for _ in range(2)]

    assert frames[0] == {
        "kind": "live_events",
        "events": events,
        "cursor": "2026-08-03T09:00:10.000000",
    }
    # The empty polls that follow send nothing; the next frame is the
    # heartbeat, not a second (empty) live_events.
    assert frames[1]["kind"] == "heartbeat"


def test_the_cursor_advances_so_the_same_event_is_never_delivered_twice(app, monkeypatch):
    seen = []
    events = [{"kind": "search", "time": "2026-08-03T09:00:10.000000", "qa_id": "a"}]
    delivered = False

    async def once(_dataset_id, since=None, user=None):
        nonlocal delivered
        seen.append(since)
        if not delivered:
            delivered = True
            return {"events": events, "cursor": "2026-08-03T09:00:10.000000"}
        return {"events": [], "cursor": since.isoformat() if since else None}

    monkeypatch.setattr(live_updates, "get_live_events", once)

    with _connect(app) as connection:
        connection.receive_json()
        while connection.receive_json()["kind"] != "heartbeat":
            pass

    assert seen[0] is None
    assert seen[1] == datetime(2026, 8, 3, 9, 0, 10)


def test_a_run_completing_is_announced_but_one_already_complete_is_not(app, monkeypatch):
    baseline_run, new_run = uuid4(), uuid4()
    reads = {"count": 0}

    async def current_run(_dataset_id, _pipeline_name):
        # The connection's baseline read and the first poll see the run that
        # was already complete; a later one sees a newer run finish.
        reads["count"] += 1
        return _completed_run(baseline_run if reads["count"] <= 2 else new_run)

    monkeypatch.setattr(live_updates, "get_pipeline_run_by_dataset", current_run)

    with _connect(app) as connection:
        connection.receive_json()
        frame = connection.receive_json()

    # The baseline run is never announced — only the one that completed while
    # the client was watching.
    assert frame == {"kind": "graph_grew", "pipeline_run_id": str(new_run)}


def test_a_run_still_in_flight_is_not_announced_as_growth(app, monkeypatch):
    started = SimpleNamespace(
        pipeline_run_id=uuid4(), status=PipelineRunStatus.DATASET_PROCESSING_STARTED
    )
    monkeypatch.setattr(
        live_updates, "get_pipeline_run_by_dataset", AsyncMock(return_value=started)
    )

    with _connect(app) as connection:
        connection.receive_json()
        assert connection.receive_json()["kind"] == "heartbeat"


def test_heartbeats_keep_arriving_on_an_otherwise_silent_stream(app):
    with _connect(app) as connection:
        connection.receive_json()
        frames = [connection.receive_json() for _ in range(2)]

    assert [frame["kind"] for frame in frames] == ["heartbeat", "heartbeat"]
    assert datetime.fromisoformat(frames[0]["time"])


def test_an_unauthenticated_connection_is_closed_with_1008(app):
    app.dependency_overrides[get_authenticated_websocket_user] = lambda: None

    with pytest.raises(WebSocketDisconnect) as closed:
        with _connect(app) as connection:
            connection.receive_json()

    assert closed.value.code == 1008


def test_a_dataset_the_caller_cannot_read_is_closed_with_1008(app, monkeypatch):
    monkeypatch.setattr(
        router_module, "get_authorized_existing_datasets", AsyncMock(return_value=[])
    )

    with pytest.raises(WebSocketDisconnect) as closed:
        with _connect(app) as connection:
            connection.receive_json()

    assert closed.value.code == 1008


def test_read_access_lost_mid_stream_closes_with_1008(app, monkeypatch):
    """Permission is re-checked on every poll, so a revoked grant ends the
    stream rather than streaming on until the client reconnects."""
    monkeypatch.setattr(
        live_updates,
        "get_live_events",
        AsyncMock(side_effect=PermissionDeniedError(message="nope")),
    )

    with pytest.raises(WebSocketDisconnect) as closed:
        with _connect(app) as connection:
            connection.receive_json()
            connection.receive_json()

    assert closed.value.code == 1008


def test_read_access_lost_before_a_graph_poll_tick_closes_with_1008(app, monkeypatch):
    """The graph_grew tick re-checks permission on its own cadence too, not
    only via the live_events tick — a revocation is caught within one
    GRAPH_POLL_TICKS window instead of waiting on the events poll."""
    monkeypatch.setattr(
        live_updates, "get_authorized_existing_datasets", AsyncMock(return_value=[])
    )

    with pytest.raises(WebSocketDisconnect) as closed:
        with _connect(app) as connection:
            connection.receive_json()
            connection.receive_json()

    assert closed.value.code == 1008


def test_an_unexpected_stream_failure_closes_with_1011_so_clients_retry(app, monkeypatch):
    monkeypatch.setattr(
        live_updates, "get_live_events", AsyncMock(side_effect=RuntimeError("db is down"))
    )

    with pytest.raises(WebSocketDisconnect) as closed:
        with _connect(app) as connection:
            connection.receive_json()
            connection.receive_json()

    assert closed.value.code == 1011
