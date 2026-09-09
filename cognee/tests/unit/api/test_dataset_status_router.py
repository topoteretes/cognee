"""Integration test for GET /v1/datasets/status (SDK-591 follow-up).

A CI bot flagged that this endpoint (and /status/progress) still returned
the raw stored PipelineRunStatus, so a crashed run read "ABANDONED" on the
activity feed but "DATASET_PROCESSING_STARTED" here. The fix routes this
endpoint through get_effective_pipeline_status_by_datasets instead of the
raw get_pipeline_status (see datasets.py / get_pipeline_status.py) — proven
here at the router level, the same way test_status_progress_router.py
proves it for the sibling /status/progress endpoint.
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


def test_status_requires_authentication(test_client):
    response = test_client.get("/api/v1/datasets/status")

    assert response.status_code in (401, 403)


def test_status_reports_abandoned_for_a_stale_run_flat_shape(authenticated_client, monkeypatch):
    """A stale STARTED row must reach the client as ABANDONED, and the
    Union[dict[str, EffectivePipelineRunStatus], ...] response model must
    actually accept the value instead of 500ing with a
    ResponseValidationError."""
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_effective_pipeline_status_by_datasets(dataset_ids, pipeline_name):
        assert dataset_ids == [dataset_id]
        assert pipeline_name == "cognify_pipeline"
        return {str(dataset_id): "ABANDONED"}

    monkeypatch.setattr(
        datasets_module,
        "get_effective_pipeline_status_by_datasets",
        _fake_get_effective_pipeline_status_by_datasets,
    )

    response = authenticated_client.get(
        "/api/v1/datasets/status", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 200
    assert response.json() == {str(dataset_id): "ABANDONED"}


def test_status_reports_abandoned_for_a_stale_run_nested_shape(authenticated_client, monkeypatch):
    """Same as above, through the nested {dataset_id: {pipeline_name: ...}}
    shape used for multiple requested pipelines — a separate code path in
    _fan_out_by_pipeline from the flat one above."""
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_effective_pipeline_status_by_datasets(dataset_ids, pipeline_name):
        return {str(dataset_id): "ABANDONED"}

    monkeypatch.setattr(
        datasets_module,
        "get_effective_pipeline_status_by_datasets",
        _fake_get_effective_pipeline_status_by_datasets,
    )

    response = authenticated_client.get(
        "/api/v1/datasets/status",
        params={"dataset": str(dataset_id), "pipeline": ["add_pipeline", "cognify_pipeline"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        str(dataset_id): {
            "add_pipeline": "ABANDONED",
            "cognify_pipeline": "ABANDONED",
        }
    }


def test_status_reports_raw_status_for_a_fresh_run(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_effective_pipeline_status_by_datasets(dataset_ids, pipeline_name):
        return {str(dataset_id): "DATASET_PROCESSING_STARTED"}

    monkeypatch.setattr(
        datasets_module,
        "get_effective_pipeline_status_by_datasets",
        _fake_get_effective_pipeline_status_by_datasets,
    )

    response = authenticated_client.get(
        "/api/v1/datasets/status", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 200
    assert response.json() == {str(dataset_id): "DATASET_PROCESSING_STARTED"}


def test_status_error_returns_409(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _raise(*args, **kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(datasets_module, "get_effective_pipeline_status_by_datasets", _raise)

    response = authenticated_client.get(
        "/api/v1/datasets/status", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 409
