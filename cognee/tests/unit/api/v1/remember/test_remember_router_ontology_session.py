"""POST /v1/remember rejects ontology_key together with session_id.

The MCP client refuses this combination before it sends the request, but a
direct HTTP caller bypasses that check entirely. An ontology grounds entity
extraction and the session-cache path never extracts, so without this the keys
are accepted and silently ignored.
"""

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


@pytest.fixture(autouse=True)
def ontology_stub(monkeypatch):
    """The keys never reach a real store here; this test is about the guard."""
    from cognee.api.v1.ontologies.ontologies import OntologyService

    monkeypatch.setattr(
        OntologyService, "get_ontology_contents", lambda self, keys, user: ["<owl/>"] * len(keys)
    )


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


def test_ontology_key_with_session_id_is_rejected(client, remember_stub):
    response = _post(client, session_id="s1", ontology_key=["organizations"])

    assert response.status_code == 400, response.text
    assert "ontology_key is only supported for permanent writes" in response.text
    # Rejected before any work is dispatched, not after a partial write.
    remember_stub.assert_not_awaited()


def test_ontology_key_alone_is_accepted(client, remember_stub):
    assert _post(client, ontology_key=["organizations"]).status_code in (200, 201)
    remember_stub.assert_awaited_once()


def test_session_id_alone_is_accepted(client, remember_stub):
    assert _post(client, session_id="s1").status_code in (200, 201)
    remember_stub.assert_awaited_once()


def test_empty_ontology_key_does_not_block_a_session_write(client, remember_stub):
    """Swagger UI submits untouched array items as "", which must not count."""
    assert _post(client, session_id="s1", ontology_key=[""]).status_code in (200, 201)
    remember_stub.assert_awaited_once()
