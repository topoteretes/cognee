"""Regression test: dataset data endpoints return 404 for missing/unauthorized datasets.

``GET /datasets/{id}/data`` and ``GET /datasets/{id}/data/{data_id}/raw`` guard
with ``if dataset is None`` and then index ``dataset[0].id``. But
``get_authorized_existing_datasets`` always returns a *list* (``[]`` when the
dataset is missing or the user lacks permission), never ``None`` — so the guard
was dead code and ``dataset[0].id`` raised ``IndexError`` (HTTP 500) instead of a
clean 404. (``GET .../data`` additionally built ``ErrorResponseDTO`` with a
positional arg, which Pydantic v2 rejects, so even the dead branch would have
500'd.) The guards now use ``if not dataset``.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture(scope="module")
def test_client():
    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/api/v1/datasets")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    import importlib

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    datasets_router_module.send_telemetry = lambda *args, **kwargs: None

    test_client.app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _no_authorized_datasets(monkeypatch):
    """Simulate a missing/unauthorized dataset: an empty list, as the real helper returns."""
    import importlib

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    monkeypatch.setattr(
        datasets_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[]),
    )


def test_get_dataset_data_missing_dataset_returns_404(client, monkeypatch):
    _no_authorized_datasets(monkeypatch)
    dataset_id = uuid.uuid4()

    response = client.get(f"/api/v1/datasets/{dataset_id}/data")

    assert response.status_code == 404
    assert "not found" in response.text.lower()


def test_get_raw_data_missing_dataset_returns_404(client, monkeypatch):
    _no_authorized_datasets(monkeypatch)
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    response = client.get(f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw")

    assert response.status_code == 404
    assert "not found" in response.text.lower()
