"""Unit tests for the /v1/remember endpoint's self_improvement form field.

remember() self-improves by default and improve() reads the whole graph, so a
deployment on a large graph needs a way to store data without paying for a
graph-wide enrichment pass on every write. The endpoint had no such switch,
which also left the MCP server's API mode unable to offer one.

Pure tests: ``remember`` itself is monkeypatched, so no DB, LLM, or network.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())


class FakeRememberResult:
    status = "completed"

    def to_dict(self):
        return {"status": "completed", "dataset_name": "test_dataset"}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


@pytest.fixture
def fake_remember(monkeypatch):
    """Capture the kwargs the router forwards to remember()."""
    import importlib

    # The package re-exports the function under its own name, so import the
    # package explicitly rather than through attribute lookup.
    remember_pkg = importlib.import_module("cognee.api.v1.remember")

    calls = []

    async def _fake_remember(data, **kwargs):
        calls.append(kwargs)
        return FakeRememberResult()

    _fake_remember.calls = calls
    monkeypatch.setattr(remember_pkg, "remember", _fake_remember)
    return _fake_remember


def _post(client, form):
    return client.post(
        "/remember",
        data=form,
        files=[("data", ("note.txt", b"a short fact", "text/plain"))],
    )


def test_self_improvement_defaults_to_true(client, fake_remember):
    resp = _post(client, {"datasetName": "test_dataset"})

    assert resp.status_code == 200
    assert fake_remember.calls[0]["self_improvement"] is True


def test_self_improvement_false_is_forwarded(client, fake_remember):
    resp = _post(client, {"datasetName": "test_dataset", "self_improvement": "false"})

    assert resp.status_code == 200
    assert fake_remember.calls[0]["self_improvement"] is False


def test_self_improvement_true_is_forwarded(client, fake_remember):
    resp = _post(client, {"datasetName": "test_dataset", "self_improvement": "true"})

    assert resp.status_code == 200
    assert fake_remember.calls[0]["self_improvement"] is True
