"""Integration test for GET /v1/datasets/status/progress (CLO-557).

/status/progress is a dedicated endpoint (not a flag on /status) so its
response shape never branches at runtime — see get_datasets_router.py's
docstring. Covers what a router-level test actually exercises: the real
auth dependency is enforced, the `dataset`/`pipeline` query aliases are
parsed by FastAPI, and cognee_datasets.get_progress's flat-vs-nested shape
reaches the client unchanged.
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
def authenticated_client(test_client, monkeypatch):
    import importlib

    router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)

    async def _override_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    test_client.app.dependency_overrides[get_authenticated_user] = _override_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _authorize_one_dataset(monkeypatch, dataset_id):
    import importlib

    router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id)]),
    )


def test_status_progress_requires_authentication(test_client):
    """No dependency override, no auth cookie/header — the real
    get_authenticated_user dependency must reject the request rather than
    the route ever running."""
    response = test_client.get("/api/v1/datasets/status/progress")

    assert response.status_code in (401, 403)


def test_status_progress_single_pipeline_flat_shape(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_pipeline_progress(dataset_ids, pipeline_name):
        assert dataset_ids == [dataset_id]
        assert pipeline_name == "cognify_pipeline"
        return {
            str(dataset_id): {
                "status": "DATASET_PROCESSING_STARTED",
                "progress": {
                    "completed_items": 3,
                    "total_items": 10,
                    "current_stage": "extract_graph_from_data",
                },
            }
        }

    monkeypatch.setattr(datasets_module, "get_pipeline_progress", _fake_get_pipeline_progress)

    response = authenticated_client.get(
        "/api/v1/datasets/status/progress", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        str(dataset_id): {
            "status": "DATASET_PROCESSING_STARTED",
            "progress": {
                "completed_items": 3,
                "total_items": 10,
                "current_stage": "extract_graph_from_data",
            },
        }
    }


def test_status_progress_multiple_pipelines_nested_shape(authenticated_client, monkeypatch):
    """The `pipeline` query alias accepts repeats; >1 distinct pipeline name
    switches the response to the nested {dataset_id: {pipeline_name: ...}} shape."""
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_pipeline_progress(dataset_ids, pipeline_name):
        return {
            str(dataset_id): {
                "status": "DATASET_PROCESSING_COMPLETED",
                "progress": None,
            }
        }

    monkeypatch.setattr(datasets_module, "get_pipeline_progress", _fake_get_pipeline_progress)

    response = authenticated_client.get(
        "/api/v1/datasets/status/progress",
        params={"dataset": str(dataset_id), "pipeline": ["add_pipeline", "cognify_pipeline"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        str(dataset_id): {
            "add_pipeline": {"status": "DATASET_PROCESSING_COMPLETED", "progress": None},
            "cognify_pipeline": {"status": "DATASET_PROCESSING_COMPLETED", "progress": None},
        }
    }


def test_status_progress_error_returns_409(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _raise(*args, **kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(datasets_module, "get_pipeline_progress", _raise)

    response = authenticated_client.get(
        "/api/v1/datasets/status/progress", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 409
