"""POST /v1/remember forwards ``self_improvement`` so API-mode clients (MCP, CLI)
can opt out of the improve loop the same way the SDK can."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


@pytest.fixture
def remember_stub(monkeypatch):
    remember_pkg = importlib.import_module("cognee.api.v1.remember")
    result = SimpleNamespace(status="completed", to_dict=lambda: {"status": "completed"})
    stub = AsyncMock(return_value=result)
    monkeypatch.setattr(remember_pkg, "remember", stub)
    return stub


def _post(client, **form):
    return client.post(
        "/remember",
        data={"datasetName": "docs", **form},
        files={"data": ("x.txt", b"hello", "text/plain")},
    )


def test_self_improvement_defaults_to_true(client, remember_stub):
    resp = _post(client)

    assert resp.status_code == 200
    assert remember_stub.call_args.kwargs["self_improvement"] is True


def test_self_improvement_false_is_forwarded(client, remember_stub):
    resp = _post(client, self_improvement="false")

    assert resp.status_code == 200
    assert remember_stub.call_args.kwargs["self_improvement"] is False


def test_self_improvement_is_documented():
    router = get_remember_router()
    route = next(r for r in router.routes if getattr(r, "path", None) == "")
    assert "self_improvement" in (route.endpoint.__doc__ or "")
