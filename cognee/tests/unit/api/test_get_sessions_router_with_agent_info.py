"""Wiring tests for GET /api/v1/sessions/with-agent-info (CLO-434).

The visibility-resolution + join/merge logic lives in
``cognee.modules.session_lifecycle.agent_usage.get_sessions_with_agent_info``
and is covered there (see ``test_agent_usage.py``). This file only checks
that the router forwards params and maps errors to HTTP — including that a
``CogneeApiError`` is left for the global handler instead of being turned
into a generic 500.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.sessions.routers import get_sessions_router as router_module
from cognee.exceptions import CogneeSystemError


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router_module.get_sessions_router(), prefix="/api/v1/sessions")
    return app


def test_with_agent_info_returns_module_result(monkeypatch):
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )

    captured_kwargs = {}
    fake_page = {
        "sessions": [{"session_id": "claude-code-123", "agent_type": "claude_code"}],
        "total": 1,
        "limit": 50,
        "offset": 0,
        "has_more": False,
    }

    async def fake_get_sessions_with_agent_info(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_page

    monkeypatch.setattr(
        router_module, "get_sessions_with_agent_info", fake_get_sessions_with_agent_info
    )

    response = TestClient(app).get(
        "/api/v1/sessions/with-agent-info", params={"limit": 10, "offset": 5}
    )

    assert response.status_code == 200
    assert response.json() == fake_page
    assert captured_kwargs["user"].id == user_id
    assert captured_kwargs["since"] is not None
    assert captured_kwargs["limit"] == 10
    assert captured_kwargs["offset"] == 5


def test_with_agent_info_failure_returns_500(monkeypatch):
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )

    async def failing(**_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(router_module, "get_sessions_with_agent_info", failing)

    response = TestClient(app).get("/api/v1/sessions/with-agent-info")
    assert response.status_code == 500
    assert response.json() == {"error": "list failed"}


def test_with_agent_info_lets_cognee_api_error_propagate(monkeypatch):
    """A CogneeApiError carries its own status/message — it must reach the
    global exception_handler in cognee/api/client.py, not get flattened
    into the generic {"error": "list failed"} 500."""
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )

    async def failing(**_kwargs):
        raise CogneeSystemError(message="boom")

    monkeypatch.setattr(router_module, "get_sessions_with_agent_info", failing)

    with pytest.raises(CogneeSystemError):
        TestClient(app).get("/api/v1/sessions/with-agent-info")
