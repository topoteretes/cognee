"""Router pass-through of the dry_run query flag on the DELETE dataset endpoints.

With ``?dry_run=true`` the route returns the structured deletion-scope report
from the datasets API instead of deleting; without the flag the call is
forwarded unchanged (no ``dry_run`` argument), preserving the pre-existing
behavior (issue #5672).
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

REPORT = {
    "status": "dry_run",
    "dry_run": True,
    "operation": "delete_all",
    "dataset_count": 1,
}


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

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    datasets_router_module.send_telemetry = lambda *args, **kwargs: None

    test_client.app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _mocked_datasets_api(**methods):
    fake = MagicMock()
    for name, mock in methods.items():
        setattr(fake, name, mock)
    return fake


def test_delete_all_dry_run_returns_report(client):
    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    delete_all = AsyncMock(return_value=REPORT)
    fake = _mocked_datasets_api(delete_all=delete_all)

    with patch.object(datasets_router_module, "datasets", fake):
        response = client.delete("/api/v1/datasets", params={"dry_run": "true"})

    assert response.status_code == 200
    assert response.json() == REPORT
    delete_all.assert_awaited_once_with(ANY, dry_run=True)


def test_delete_dataset_dry_run_returns_report(client):
    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    report = {
        "status": "dry_run",
        "dry_run": True,
        "operation": "empty_dataset",
        "data_count": 2,
    }
    empty_dataset = AsyncMock(return_value=report)
    fake = _mocked_datasets_api(empty_dataset=empty_dataset)
    dataset_id = str(uuid.uuid4())

    with patch.object(datasets_router_module, "datasets", fake):
        response = client.delete(f"/api/v1/datasets/{dataset_id}", params={"dry_run": "true"})

    assert response.status_code == 200
    assert response.json() == report
    empty_dataset.assert_awaited_once_with(uuid.UUID(dataset_id), ANY, dry_run=True)


def test_delete_data_dry_run_returns_report(client):
    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    report = {
        "status": "dry_run",
        "dry_run": True,
        "operation": "delete_data",
        "tracked_data_row": False,
    }
    delete_data = AsyncMock(return_value=report)
    fake = _mocked_datasets_api(delete_data=delete_data)
    dataset_id = str(uuid.uuid4())
    data_id = str(uuid.uuid4())

    with patch.object(datasets_router_module, "datasets", fake):
        response = client.delete(
            f"/api/v1/datasets/{dataset_id}/data/{data_id}", params={"dry_run": "true"}
        )

    assert response.status_code == 200
    assert response.json() == report
    delete_data.assert_awaited_once_with(
        uuid.UUID(dataset_id), uuid.UUID(data_id), ANY, dry_run=True
    )


def test_delete_dataset_without_dry_run_still_deletes(client):
    """Regression: no query flag means the plain destructive call, no report."""
    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    empty_dataset = AsyncMock(return_value=None)
    fake = _mocked_datasets_api(empty_dataset=empty_dataset)
    dataset_id = str(uuid.uuid4())

    with patch.object(datasets_router_module, "datasets", fake):
        response = client.delete(f"/api/v1/datasets/{dataset_id}")

    assert response.status_code == 200
    assert response.json() is None
    empty_dataset.assert_awaited_once_with(uuid.UUID(dataset_id), ANY)
