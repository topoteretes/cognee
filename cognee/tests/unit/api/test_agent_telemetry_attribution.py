"""An agent authenticating over HTTP is attributed in every event of its request.

Runs the real ``get_authenticated_user`` (only the credential check beneath it
is overridden), so this guards the hook that binds the agent, and that the
binding reaches the handler, background tasks the handler starts, and the
error telemetry — including the unhandled-500 middleware, which runs outside
the handler's context and only sees the agent through ``request.state``.
"""

import asyncio
import importlib
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.api.exception_telemetry import API_EXCEPTION_EVENT, send_api_exception_telemetry
from cognee.exceptions import CogneeApiError
from cognee.shared import utils
from cognee.shared.utils import send_telemetry

# The methods package re-exports the function under the module's own name.
auth_module = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")


class FakeUser:
    def __init__(self, parent_user_id=None, email=None):
        self.id = uuid.uuid4()
        self.tenant_id = None
        self.parent_user_id = parent_user_id
        self.email = email


def _app_for(user, monkeypatch) -> tuple[FastAPI, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]):
        # Recorded when send_telemetry builds the event, not when its
        # fire-and-forget task runs, so events raised after the response
        # (the exception middleware) are never missed.
        payloads.append(payload)

        async def noop() -> None:
            return None

        return noop()

    async def get_user(_user_id):
        return user

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils, "_send_telemetry_request", capture)
    monkeypatch.setattr(auth_module, "get_user", get_user)

    app = FastAPI()
    app.dependency_overrides[auth_module._auth_dependency] = lambda: user

    # Wired like cognee/api/client.py.
    @app.middleware("http")
    async def _report_unhandled_exceptions(request, call_next):
        try:
            return await call_next(request)
        except Exception as error:
            send_api_exception_telemetry(request, error, status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise

    @app.exception_handler(CogneeApiError)
    async def _cognee_error(request: Request, exc: CogneeApiError):
        send_api_exception_telemetry(request, exc, exc.status_code, error_name=exc.name)
        return JSONResponse(status_code=exc.status_code, content={"detail": "error"})

    @app.post("/crash")
    async def crash(_user=Depends(auth_module.get_authenticated_user)):
        raise RuntimeError("unhandled")

    @app.post("/handled")
    async def handled(_user=Depends(auth_module.get_authenticated_user)):
        raise CogneeApiError(message="handled", name="HandledError", status_code=409, log=False)

    @app.post("/work")
    async def work(_user=Depends(auth_module.get_authenticated_user)):
        send_telemetry("handler event", "sdk")

        async def background():
            send_telemetry("background event", "sdk")

        await asyncio.create_task(background())
        return {}

    return app, payloads


def test_agent_requests_attribute_handler_and_background_events(monkeypatch):
    parent_id = uuid.uuid4()
    agent = FakeUser(parent_id, f"opencode+{parent_id}@cognee.agent")
    app, payloads = _app_for(agent, monkeypatch)

    assert TestClient(app).post("/work").status_code == 200

    assert {payload["event_name"] for payload in payloads} == {
        "handler event",
        "background event",
    }
    for payload in payloads:
        assert payload["properties"]["agent_id"] == str(agent.id)
        assert payload["properties"]["agent_owner_id"] == str(parent_id)
        assert payload["properties"]["agent_plugin"] == "opencode"


def test_human_requests_carry_no_agent(monkeypatch):
    app, payloads = _app_for(FakeUser(), monkeypatch)

    assert TestClient(app).post("/work").status_code == 200

    assert payloads
    assert all(payload["properties"]["is_agent"] is False for payload in payloads)


def _agent() -> FakeUser:
    parent_id = uuid.uuid4()
    return FakeUser(parent_id, f"codex+{parent_id}@cognee.agent")


def _exception_events(payloads):
    return [payload for payload in payloads if payload["event_name"] == API_EXCEPTION_EVENT]


def test_an_agents_unhandled_500_keeps_the_agent(monkeypatch):
    """The exception middleware sits outside the handler's context (review finding)."""
    agent = _agent()
    app, payloads = _app_for(agent, monkeypatch)

    response = TestClient(app, raise_server_exceptions=False).post("/crash")

    assert response.status_code == 500
    [event] = _exception_events(payloads)
    assert event["properties"]["status_code"] == 500
    assert event["properties"]["agent_id"] == str(agent.id)
    assert event["properties"]["agent_plugin"] == "codex"


def test_an_agents_handled_error_keeps_the_agent(monkeypatch):
    agent = _agent()
    app, payloads = _app_for(agent, monkeypatch)

    assert TestClient(app).post("/handled").status_code == 409

    [event] = _exception_events(payloads)
    assert event["properties"]["agent_id"] == str(agent.id)


def test_a_humans_unhandled_500_carries_no_agent(monkeypatch):
    app, payloads = _app_for(FakeUser(), monkeypatch)

    TestClient(app, raise_server_exceptions=False).post("/crash")

    [event] = _exception_events(payloads)
    assert event["properties"]["is_agent"] is False
