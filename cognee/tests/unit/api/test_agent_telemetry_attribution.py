"""API error telemetry carries the authenticated caller, agents included.

The error helper sends no ``User`` of its own: it reads the one
``get_authenticated_user`` stored on ``request.state``. That matters most for
the unhandled-500 middleware, which runs outside the handler's context and so
cannot see the dependency's result any other way. Runs the real
``get_authenticated_user`` (only the credential check beneath it is
overridden), so this guards the line that stores the caller.
"""

import importlib
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.api.exception_telemetry import API_EXCEPTION_EVENT, send_api_exception_telemetry
from cognee.exceptions import CogneeApiError
from cognee.shared import utils

# The methods package re-exports the function under the module's own name.
auth_module = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")


class FakeUser:
    def __init__(self, parent_user_id=None, email=None):
        self.id = uuid.uuid4()
        self.tenant_id = uuid.uuid4()
        self.parent_user_id = parent_user_id
        self.email = email


def _agent() -> FakeUser:
    parent_id = uuid.uuid4()
    return FakeUser(parent_id, f"codex+{parent_id}@cognee.agent")


def _app_for(user, monkeypatch) -> tuple[FastAPI, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    async def get_user(_user_id):
        return user

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    # Recorded when send_telemetry queues the event, so events raised after the
    # response (the exception middleware) are never missed.
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", payloads.append)
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

    @app.post("/unauthenticated-crash")
    async def unauthenticated_crash():
        raise RuntimeError("unhandled")

    return app, payloads


def _exception_event(payloads) -> dict[str, Any]:
    [event] = [payload for payload in payloads if payload["event_name"] == API_EXCEPTION_EVENT]
    return event["properties"]


def test_an_agents_unhandled_500_carries_the_agent(monkeypatch):
    """The exception middleware sits outside the handler's context (review finding)."""
    agent = _agent()
    app, payloads = _app_for(agent, monkeypatch)

    response = TestClient(app, raise_server_exceptions=False).post("/crash")

    assert response.status_code == 500
    properties = _exception_event(payloads)
    assert properties["status_code"] == 500
    assert properties["user_id"] == str(agent.id)
    assert properties["tenant_id"] == str(agent.tenant_id)
    assert properties["is_agent"] is True
    assert properties["agent_owner_id"] == str(agent.parent_user_id)
    assert properties["agent_plugin"] == "codex"


def test_an_agents_handled_error_carries_the_agent(monkeypatch):
    agent = _agent()
    app, payloads = _app_for(agent, monkeypatch)

    assert TestClient(app).post("/handled").status_code == 409

    properties = _exception_event(payloads)
    assert properties["user_id"] == str(agent.id)
    assert properties["agent_plugin"] == "codex"


def test_a_humans_unhandled_500_carries_the_human_and_no_agent(monkeypatch):
    human = FakeUser()
    app, payloads = _app_for(human, monkeypatch)

    TestClient(app, raise_server_exceptions=False).post("/crash")

    properties = _exception_event(payloads)
    assert properties["user_id"] == str(human.id)
    assert properties["is_agent"] is False


def test_an_error_before_authentication_still_reports(monkeypatch):
    """No caller was stored, so the event carries none — and nothing breaks."""
    app, payloads = _app_for(_agent(), monkeypatch)

    response = TestClient(app, raise_server_exceptions=False).post("/unauthenticated-crash")

    assert response.status_code == 500
    properties = _exception_event(payloads)
    assert properties["user_id"] == "None"
    assert properties["is_agent"] is False
