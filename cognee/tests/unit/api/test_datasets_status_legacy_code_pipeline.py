"""Deprecated ``code_graph_pipeline`` alias on the dataset status endpoints (SDK-793).

Code repositories are built under ``cognify_pipeline`` now, so legacy pollers that
still ask for ``code_graph_pipeline`` (cognee-mcp's client) must see cognify's state
for datasets with no run of their own under that name, keyed as they asked. A dataset
that does have a ``code_graph_pipeline`` run (a custom pipeline using that name)
keeps reporting its own run.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

ROUTER_MODULE = "cognee.api.v1.datasets.routers.get_datasets_router"
DATASETS_MODULE = "cognee.api.v1.datasets.datasets"


@pytest.fixture(scope="module")
def test_client():
    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/api/v1/datasets")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client, monkeypatch):
    router_module = importlib.import_module(ROUTER_MODULE)
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)

    async def _override_user():
        return SimpleNamespace(id=str(uuid.uuid4()), is_active=True, tenant_id=None)

    test_client.app.dependency_overrides[get_authenticated_user] = _override_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _authorize(monkeypatch, dataset_ids):
    router_module = importlib.import_module(ROUTER_MODULE)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id) for dataset_id in dataset_ids]),
    )


def _fake_runs(monkeypatch, runs):
    """``runs`` maps pipeline name -> {dataset_id: status}; stands in for the
    latest-run lookup behind get_status / get_progress / the alias resolver."""
    datasets_module = importlib.import_module(DATASETS_MODULE)
    calls = []

    async def fake_status(dataset_ids, pipeline_name):
        calls.append((list(dataset_ids), pipeline_name))
        by_dataset = runs.get(pipeline_name, {})
        return {str(d): by_dataset[d] for d in dataset_ids if d in by_dataset}

    async def fake_progress(dataset_ids, pipeline_name):
        statuses = await fake_status(dataset_ids, pipeline_name)
        return {d: {"status": s, "progress": None} for d, s in statuses.items()}

    monkeypatch.setattr(datasets_module, "get_pipeline_status", fake_status)
    monkeypatch.setattr(datasets_module, "get_pipeline_progress", fake_progress)
    return calls


def _capture_warnings(monkeypatch):
    datasets_module = importlib.import_module(DATASETS_MODULE)
    warnings = []
    monkeypatch.setattr(
        datasets_module.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    return warnings


def test_alias_falls_back_to_cognify_when_no_code_graph_run(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {"cognify_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"}})
    warnings = _capture_warnings(monkeypatch)

    response = client.get(
        "/api/v1/datasets/status",
        params={"dataset": str(dataset_id), "pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    assert response.json() == {str(dataset_id): "DATASET_PROCESSING_COMPLETED"}
    assert len(warnings) == 1


def test_alias_reports_own_run_when_code_graph_run_exists(client, monkeypatch):
    with_run, without_run = uuid.uuid4(), uuid.uuid4()
    _authorize(monkeypatch, [with_run, without_run])
    _fake_runs(
        monkeypatch,
        {
            "code_graph_pipeline": {with_run: "DATASET_PROCESSING_STARTED"},
            "cognify_pipeline": {
                with_run: "DATASET_PROCESSING_COMPLETED",
                without_run: "DATASET_PROCESSING_ERRORED",
            },
        },
    )

    response = client.get(
        "/api/v1/datasets/status",
        params={"dataset": [str(with_run), str(without_run)], "pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    assert response.json() == {
        str(with_run): "DATASET_PROCESSING_STARTED",
        str(without_run): "DATASET_PROCESSING_ERRORED",
    }


def test_alias_with_no_runs_at_all_stays_empty_without_warning(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {})
    warnings = _capture_warnings(monkeypatch)

    response = client.get(
        "/api/v1/datasets/status",
        params={"dataset": str(dataset_id), "pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    assert response.json() == {}
    assert warnings == []


def test_multi_pipeline_shape_keeps_requested_key(client, monkeypatch):
    """cognee-mcp's poller: nested {dataset: {pipeline: status}} with every key it asked for."""
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(
        monkeypatch,
        {
            "add_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"},
            "cognify_pipeline": {dataset_id: "DATASET_PROCESSING_STARTED"},
        },
    )
    warnings = _capture_warnings(monkeypatch)

    response = client.get(
        "/api/v1/datasets/status",
        params={
            "dataset": str(dataset_id),
            "pipeline": ["add_pipeline", "cognify_pipeline", "code_graph_pipeline"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        str(dataset_id): {
            "add_pipeline": "DATASET_PROCESSING_COMPLETED",
            "cognify_pipeline": "DATASET_PROCESSING_STARTED",
            "code_graph_pipeline": "DATASET_PROCESSING_STARTED",
        }
    }
    assert len(warnings) == 1


def test_other_custom_pipeline_names_do_not_fall_back(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    calls = _fake_runs(
        monkeypatch, {"cognify_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"}}
    )

    response = client.get(
        "/api/v1/datasets/status", params={"dataset": str(dataset_id), "pipeline": "my_pipeline"}
    )

    assert response.status_code == 200
    assert response.json() == {}
    assert [name for _, name in calls] == ["my_pipeline"]


def test_progress_endpoint_applies_the_same_fallback(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {"cognify_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"}})

    response = client.get(
        "/api/v1/datasets/status/progress",
        params={"dataset": str(dataset_id), "pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    assert response.json() == {
        str(dataset_id): {"status": "DATASET_PROCESSING_COMPLETED", "progress": None}
    }


def _patch_count_helper(monkeypatch):
    methods_module = importlib.import_module("cognee.modules.data.methods")
    helper = AsyncMock(return_value={"total": 0, "completed": 0, "pending": 0, "items": []})
    monkeypatch.setattr(methods_module, "get_dataset_processing_status", helper)
    return helper


def test_processing_status_alias_counts_cognify_without_code_graph_run(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {"cognify_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"}})
    warnings = _capture_warnings(monkeypatch)
    helper = _patch_count_helper(monkeypatch)

    response = client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    helper.assert_awaited_once_with(dataset_id, pipeline_name="cognify_pipeline")
    assert len(warnings) == 1


def test_processing_status_alias_counts_own_run_when_present(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {"code_graph_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"}})
    helper = _patch_count_helper(monkeypatch)

    response = client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    helper.assert_awaited_once_with(dataset_id, pipeline_name="code_graph_pipeline")
