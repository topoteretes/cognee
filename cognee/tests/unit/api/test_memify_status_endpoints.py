from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture(scope="session")
def test_client():
    from cognee.api.v1.memify.routers.get_memify_router import get_memify_router

    app = FastAPI()
    app.include_router(get_memify_router(), prefix="/api/v1/memify")

    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid4()),
        )

    import importlib

    memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")
    memify_router_module.send_telemetry = lambda *args, **kwargs: None

    test_client.app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _build_pipeline_run(*, dataset_id, pipeline_name="memify_pipeline", dataset_name="dataset"):
    return SimpleNamespace(
        pipeline_run_id=uuid4(),
        dataset_id=dataset_id,
        pipeline_name=pipeline_name,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        run_info={"status": "PipelineRunStarted", "dataset_name": dataset_name},
    )


def test_get_latest_memify_status_by_dataset_name(client, monkeypatch):
    import importlib

    dataset_id = uuid4()
    pipeline_run = _build_pipeline_run(dataset_id=dataset_id, dataset_name="research-notes")

    memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")
    monkeypatch.setattr(
        memify_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id, name="research-notes")]),
    )
    monkeypatch.setattr(
        memify_router_module,
        "get_pipeline_run_by_dataset",
        AsyncMock(return_value=pipeline_run),
    )

    response = client.get("/api/v1/memify/status", params={"dataset_name": "research-notes"})

    assert response.status_code == 200
    assert response.json() == {
        "pipeline_run_id": str(pipeline_run.pipeline_run_id),
        "dataset_id": str(dataset_id),
        "pipeline_name": "memify_pipeline",
        "status": "DATASET_PROCESSING_STARTED",
        "created_at": "2025-01-01T00:00:00Z",
        "run_info": {"status": "PipelineRunStarted", "dataset_name": "research-notes"},
        "dataset_name": "research-notes",
    }


def test_get_latest_memify_status_strips_raw_data_from_run_info(client, monkeypatch):
    import importlib

    dataset_id = uuid4()
    pipeline_run = SimpleNamespace(
        pipeline_run_id=uuid4(),
        dataset_id=dataset_id,
        pipeline_name="memify_pipeline",
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        run_info={"data": "secret note body", "error": "pipeline failed"},
    )

    memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")
    monkeypatch.setattr(
        memify_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id, name="research-notes")]),
    )
    monkeypatch.setattr(
        memify_router_module,
        "get_pipeline_run_by_dataset",
        AsyncMock(return_value=pipeline_run),
    )

    response = client.get("/api/v1/memify/status", params={"dataset_name": "research-notes"})

    assert response.status_code == 200
    assert response.json()["run_info"] == {"error": "pipeline failed"}
    assert response.json()["dataset_name"] == "research-notes"


def test_get_latest_memify_status_rejects_ambiguous_dataset_name(client, monkeypatch):
    import importlib

    dataset_id = uuid4()
    memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")
    monkeypatch.setattr(
        memify_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(
            return_value=[
                SimpleNamespace(id=dataset_id, name="research-notes"),
                SimpleNamespace(id=uuid4(), name="research-notes"),
            ]
        ),
    )

    response = client.get("/api/v1/memify/status", params={"dataset_name": "research-notes"})

    assert response.status_code == 409
    assert (
        response.json()["error"]
        == "Multiple readable datasets match dataset_name. Use dataset_id instead."
    )


def test_get_latest_memify_status_requires_exactly_one_dataset_selector(client):
    response = client.get("/api/v1/memify/status")
    assert response.status_code == 400
    assert response.json()["error"] == "Provide exactly one of dataset_name or dataset_id."

    response = client.get(
        "/api/v1/memify/status",
        params={"dataset_name": "research-notes", "dataset_id": str(uuid4())},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Provide exactly one of dataset_name or dataset_id."


def test_get_memify_status_by_run_id_returns_404_for_non_memify_runs(client, monkeypatch):
    import importlib

    dataset_id = uuid4()
    pipeline_run = _build_pipeline_run(dataset_id=dataset_id, pipeline_name="cognify_pipeline")

    memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")
    monkeypatch.setattr(
        memify_router_module,
        "get_pipeline_run",
        AsyncMock(return_value=pipeline_run),
    )

    response = client.get(f"/api/v1/memify/status/{pipeline_run.pipeline_run_id}")

    assert response.status_code == 404
    assert response.json()["error"] == "Memify pipeline run not found."


def test_get_memify_status_by_run_id_checks_dataset_access(client, monkeypatch):
    import importlib

    dataset_id = uuid4()
    pipeline_run = _build_pipeline_run(dataset_id=dataset_id)

    memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")
    monkeypatch.setattr(
        memify_router_module,
        "get_pipeline_run",
        AsyncMock(return_value=pipeline_run),
    )
    monkeypatch.setattr(
        memify_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[]),
    )

    response = client.get(f"/api/v1/memify/status/{pipeline_run.pipeline_run_id}")

    assert response.status_code == 404
    assert response.json()["error"] == "Memify pipeline run not found."