"""Wiring tests for GET /api/v1/sessions/cost-by-user-agent (CLO-434 follow-up).

The aggregation/role-scoping logic lives in
``cognee.modules.session_lifecycle.agent_usage`` and is covered there
(see ``test_agent_usage.py``). This file only checks that the router
forwards the range filter and maps errors to HTTP.
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


def test_cost_by_user_agent_returns_module_result(monkeypatch):
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None, email="me@example.com"
    )

    captured = {}
    fake_result = [{"user_id": str(user_id), "user_email": "me@example.com", "agent_type": "codex"}]

    async def fake_compute(**kwargs):
        captured.update(kwargs)
        return fake_result

    monkeypatch.setattr(router_module, "compute_cost_by_user_agent", fake_compute)

    response = TestClient(app).get("/api/v1/sessions/cost-by-user-agent", params={"range": "7d"})

    assert response.status_code == 200
    assert response.json() == fake_result
    assert captured["user"].id == user_id
    assert captured["since"] is not None


def test_cost_by_user_agent_failure_returns_500(monkeypatch):
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None, email="me@example.com"
    )

    async def failing_compute(**_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(router_module, "compute_cost_by_user_agent", failing_compute)

    response = TestClient(app).get("/api/v1/sessions/cost-by-user-agent")
    assert response.status_code == 500
    assert response.json() == {"error": "aggregation failed"}
