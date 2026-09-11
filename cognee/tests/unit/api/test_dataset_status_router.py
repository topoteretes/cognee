"""Integration test for GET /v1/datasets/status (SDK-591).

This endpoint reports the STORED PipelineRunStatus, so a crashed run reads
"ABANDONED" on the activity feed and "DATASET_PROCESSING_STARTED" here. The
divergence is deliberate and these tests exist to keep it.

Every frontend status path reads this endpoint through one shared mapper
(mapProcessingStatus in cognee-frontend/src/app/(app)/datasets/brainsTypes.ts),
which falls through to "completed" for any value it does not recognise, and
through a poller whose terminal/in-progress sets do not contain ABANDONED.
Surfacing ABANDONED here would therefore show an abandoned dataset as
successfully processed, and hang upload polling until its timeout. Reporting
a dead run as still running is also wrong, but it never claims success.

The effective status is carried by /activity. This endpoint follows once the
frontend has a mapping for it, which has to start in cognee-saas-frontend
since cognee-frontend here is synced from it.
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


def test_status_reports_the_stored_status_for_a_stale_run_flat_shape(
    authenticated_client, monkeypatch
):
    """A stale STARTED row reaches the client as DATASET_PROCESSING_STARTED,
    not ABANDONED. The frontend mapper has no branch for ABANDONED and falls
    through to "completed", so surfacing it here would report a dead run as a
    successful one."""
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_pipeline_status(dataset_ids, pipeline_name):
        assert dataset_ids == [dataset_id]
        assert pipeline_name == "cognify_pipeline"
        return {str(dataset_id): "DATASET_PROCESSING_STARTED"}

    monkeypatch.setattr(
        datasets_module,
        "get_pipeline_status",
        _fake_get_pipeline_status,
    )

    response = authenticated_client.get(
        "/api/v1/datasets/status", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 200
    assert response.json() == {str(dataset_id): "DATASET_PROCESSING_STARTED"}


def test_status_reports_the_stored_status_for_a_stale_run_nested_shape(
    authenticated_client, monkeypatch
):
    """Same as above, through the nested {dataset_id: {pipeline_name: ...}}
    shape used for multiple requested pipelines — a separate code path in
    _fan_out_by_pipeline from the flat one above."""
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_pipeline_status(dataset_ids, pipeline_name):
        return {str(dataset_id): "DATASET_PROCESSING_STARTED"}

    monkeypatch.setattr(
        datasets_module,
        "get_pipeline_status",
        _fake_get_pipeline_status,
    )

    response = authenticated_client.get(
        "/api/v1/datasets/status",
        params={"dataset": str(dataset_id), "pipeline": ["add_pipeline", "cognify_pipeline"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        str(dataset_id): {
            "add_pipeline": "DATASET_PROCESSING_STARTED",
            "cognify_pipeline": "DATASET_PROCESSING_STARTED",
        }
    }


def test_status_reports_raw_status_for_a_fresh_run(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_one_dataset(monkeypatch, dataset_id)

    import importlib

    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    async def _fake_get_pipeline_status(dataset_ids, pipeline_name):
        return {str(dataset_id): "DATASET_PROCESSING_STARTED"}

    monkeypatch.setattr(
        datasets_module,
        "get_pipeline_status",
        _fake_get_pipeline_status,
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

    monkeypatch.setattr(datasets_module, "get_pipeline_status", _raise)

    response = authenticated_client.get(
        "/api/v1/datasets/status", params={"dataset": str(dataset_id)}
    )

    assert response.status_code == 409
