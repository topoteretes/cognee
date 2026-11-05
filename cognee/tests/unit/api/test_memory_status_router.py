"""GET /v1/datasets/status/memory: the folded memory state per dataset (SDK-397)."""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.cognify.memory_status import MemoryStatus
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture(scope="module")
def test_client():
    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/api/v1/datasets")
    with TestClient(app) as client:
        yield client


@pytest.fixture
def authenticated_client(test_client, monkeypatch):
    router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)

    async def _override_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()), email="u@example.com", is_active=True, tenant_id=str(uuid.uuid4())
        )

    test_client.app.dependency_overrides[get_authenticated_user] = _override_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def test_memory_status_requires_authentication(test_client):
    assert test_client.get("/api/v1/datasets/status/memory").status_code in (401, 403)


def test_memory_status_returns_one_entry_per_authorized_dataset(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id)]),
    )
    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
    status = MemoryStatus(
        state="extracting",
        search_level="chunks",
        message="Entities and relationships are being extracted.",
        indexing_status="DATASET_PROCESSING_COMPLETED",
        cognify_status="DATASET_PROCESSING_STARTED",
    )
    fetch = AsyncMock(return_value={str(dataset_id): status})
    monkeypatch.setattr(datasets_module.datasets, "get_memory_status", fetch)

    response = authenticated_client.get(
        "/api/v1/datasets/status/memory", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 200, response.text
    fetch.assert_awaited_once_with([dataset_id])
    body = response.json()[str(dataset_id)]
    assert body["state"] == "extracting"
    assert body["search_level"] == "chunks"
    assert body["searchable"] is True and body["graph_ready"] is False
    assert body["cognify_status"] == "DATASET_PROCESSING_STARTED"
    assert body["error_class"] is None
