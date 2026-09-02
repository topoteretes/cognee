"""WS /cognify/subscribe/{pipeline_run_id}: it authenticates, and it guards.

The route used to call get_authenticated_user with keyword arguments that
function does not accept, so every connection raised TypeError inside its own
try/except and was closed 1008 — nobody could subscribe to a run at all, which
means nothing past that check had ever run in production. Fixing the auth makes
the rest of the route reachable for the first time, and the rest of the route
takes a caller-supplied run id straight to a process-global queue.

Subscribing *consumes* that queue (`get_from_queue` pops, `initialize_queue`
replaces it wholesale), so reaching it for someone else's run is not a read of
another tenant's data — it is taking their updates away from them. Hence the
guards these pin: a valid id, a run that exists, and a run whose dataset the
caller may read, all settled before the queue is touched at all.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_websocket_user

router_module = import_module("cognee.api.v1.cognify.routers.get_cognify_router")

PIPELINE_RUN_ID = uuid4()
DATASET_ID = uuid4()


@pytest.fixture
def queue_calls(monkeypatch):
    """Records every queue operation the route performs.

    A refused caller must leave this empty: that, not the close code alone, is
    what proves the rightful subscriber's updates were left alone.
    """
    calls = []
    completed = PipelineRunCompleted(
        pipeline_run_id=PIPELINE_RUN_ID, dataset_id=DATASET_ID, dataset_name="billing"
    )

    def _record(name, result=None):
        def _call(run_id):
            calls.append((name, run_id))
            return result

        return _call

    monkeypatch.setattr(router_module, "initialize_queue", _record("initialize"))
    monkeypatch.setattr(router_module, "remove_queue", _record("remove"))
    monkeypatch.setattr(router_module, "get_from_queue", _record("get", result=completed))
    return calls


@pytest.fixture
def app(monkeypatch, queue_calls):
    monkeypatch.setattr(
        router_module,
        "get_pipeline_run",
        AsyncMock(return_value=SimpleNamespace(dataset_id=DATASET_ID)),
    )
    monkeypatch.setattr(
        router_module,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(id=DATASET_ID)),
    )
    monkeypatch.setattr(
        router_module, "get_formatted_graph_data", AsyncMock(return_value={"nodes": []})
    )

    application = FastAPI()
    application.include_router(router_module.get_cognify_router(), prefix="/api/v1/cognify")
    application.dependency_overrides[get_authenticated_websocket_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return application


def _connect(app, run_id=PIPELINE_RUN_ID):
    return TestClient(app).websocket_connect(f"/api/v1/cognify/subscribe/{run_id}")


def _closed_code(app, run_id=PIPELINE_RUN_ID):
    with pytest.raises(WebSocketDisconnect) as closed:
        with _connect(app, run_id) as connection:
            connection.receive_json()
    return closed.value.code


def test_an_authenticated_subscriber_receives_pipeline_run_info(app):
    with _connect(app) as connection:
        assert connection.receive_json() == {
            "pipeline_run_id": str(PIPELINE_RUN_ID),
            "status": "PipelineRunCompleted",
            "payload": {"nodes": []},
        }


def test_an_unauthenticated_subscriber_is_still_closed_with_1008(app, queue_calls):
    app.dependency_overrides[get_authenticated_websocket_user] = lambda: None

    assert _closed_code(app) == 1008
    assert queue_calls == []


def test_a_run_id_that_is_not_a_uuid_is_refused_rather_than_crashing(app, queue_calls):
    """Previously an unhandled ValueError after accept(), i.e. an ASGI error."""
    assert _closed_code(app, "not-a-uuid") == 1008
    assert queue_calls == []


def test_an_unknown_run_is_refused_rather_than_dereferencing_none(app, monkeypatch, queue_calls):
    """get_pipeline_run is a scalar read: no row means None, and the route used
    to take .dataset_id straight off it."""
    monkeypatch.setattr(router_module, "get_pipeline_run", AsyncMock(return_value=None))

    assert _closed_code(app) == 1008
    assert queue_calls == []


def test_another_tenants_run_is_refused_before_its_queue_is_touched(app, monkeypatch, queue_calls):
    """The theft this guard exists for: the intruder is closed out, and the
    rightful subscriber's queue is neither reset nor drained."""
    monkeypatch.setattr(router_module, "get_authorized_dataset", AsyncMock(return_value=None))

    assert _closed_code(app) == 1008
    assert queue_calls == []


def test_losing_read_access_mid_stream_closes_instead_of_raising(app, monkeypatch):
    """get_formatted_graph_data re-authorizes on every frame and raises when
    the dataset goes away; only WebSocketDisconnect used to be caught."""
    monkeypatch.setattr(
        router_module,
        "get_formatted_graph_data",
        AsyncMock(side_effect=DatasetNotFoundError(message="Dataset not found.")),
    )

    assert _closed_code(app) == 1008
