"""GET and POST /datasets: an internal failure returns a generic 500.

The caught exception's text (a database host, name or SQL fragment when the
relational store is unavailable) must reach the server log only, never the
response body (gh #5590).
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")

# Stands in for what a driver error carries; it must not come back to the caller.
SECRET = "postgresql://cognee@db.internal:5432/cognee_db"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    app = FastAPI()
    app.include_router(router_module.get_datasets_router(), prefix="/api/v1/datasets")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), email="user@example.com", is_active=True, tenant_id=None
    )
    return TestClient(app, raise_server_exceptions=False)


def test_get_datasets_failure_hides_the_exception_text(client, monkeypatch):
    monkeypatch.setattr(
        router_module,
        "get_all_user_permission_datasets",
        AsyncMock(side_effect=RuntimeError(f"connection failed: {SECRET}")),
    )

    response = client.get("/api/v1/datasets")

    assert response.status_code == 500
    assert response.json() == {"detail": "Error retrieving datasets."}
    assert SECRET not in response.text


def test_create_dataset_failure_hides_the_exception_text(client, monkeypatch):
    monkeypatch.setattr(
        router_module,
        "get_datasets_by_name",
        AsyncMock(side_effect=RuntimeError(f"connection failed: {SECRET}")),
    )

    response = client.post("/api/v1/datasets", json={"name": "billing"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Error creating dataset."}
    assert SECRET not in response.text


def test_failure_is_logged_with_its_traceback(client, monkeypatch):
    error = RuntimeError(f"connection failed: {SECRET}")
    monkeypatch.setattr(
        router_module, "get_all_user_permission_datasets", AsyncMock(side_effect=error)
    )
    active = []

    def record_exception(message):
        # logger.exception() picks the traceback up from the active exception,
        # so the handler must call it while the failure is still being handled.
        active.append((message, sys.exc_info()[1]))

    monkeypatch.setattr(router_module.logger, "exception", record_exception)

    client.get("/api/v1/datasets")

    assert active == [("Error retrieving datasets", error)]
