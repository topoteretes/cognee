"""Wiring tests for GET /api/v1/sessions/with-agent-info (CLO-434).

The join/merge logic itself lives in
``cognee.modules.session_lifecycle.agent_usage`` and is covered there
(see ``test_agent_usage.py``). This file only checks that the router
resolves auth/visibility, forwards params, and maps errors to HTTP.
"""

from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.sessions.routers import get_sessions_router as router_module


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

    async def fake_permitted(_user):
        return ["dataset-1"]

    async def fake_visible(_user):
        return [user_id]

    async def fake_build_page(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_page

    monkeypatch.setattr(router_module, "_permitted_dataset_ids_for", fake_permitted)
    monkeypatch.setattr(router_module, "_visible_user_ids", fake_visible)
    monkeypatch.setattr(router_module, "build_sessions_with_agent_info_page", fake_build_page)

    response = TestClient(app).get(
        "/api/v1/sessions/with-agent-info", params={"limit": 10, "offset": 5}
    )

    assert response.status_code == 200
    assert response.json() == fake_page
    assert captured_kwargs["visible_user_ids"] == [user_id]
    assert captured_kwargs["permitted_dataset_ids"] == ["dataset-1"]
    assert captured_kwargs["limit"] == 10
    assert captured_kwargs["offset"] == 5


def test_with_agent_info_failure_returns_500(monkeypatch):
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )

    async def fake_permitted(_user):
        return []

    async def fake_visible(_user):
        return [user_id]

    async def failing_build_page(**_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(router_module, "_permitted_dataset_ids_for", fake_permitted)
    monkeypatch.setattr(router_module, "_visible_user_ids", fake_visible)
    monkeypatch.setattr(router_module, "build_sessions_with_agent_info_page", failing_build_page)

    response = TestClient(app).get("/api/v1/sessions/with-agent-info")
    assert response.status_code == 500
    assert response.json() == {"error": "list failed"}
