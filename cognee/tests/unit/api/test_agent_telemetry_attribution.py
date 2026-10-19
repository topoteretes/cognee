"""An agent authenticating over HTTP is attributed in every event of its request.

Runs the real ``get_authenticated_user`` (only the credential check beneath it
is overridden), so this guards the hook that binds the agent, and that the
binding reaches the handler and background tasks the handler starts.
"""

import asyncio
import importlib
import uuid
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

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

    async def capture(payload: dict[str, Any]) -> None:
        payloads.append(payload)

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

    @app.post("/work")
    async def work(_user=Depends(auth_module.get_authenticated_user)):
        send_telemetry("handler event", "sdk")

        async def background():
            send_telemetry("background event", "sdk")

        await asyncio.create_task(background())
        # Let the fire-and-forget telemetry tasks run before the response.
        await asyncio.sleep(0)
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
