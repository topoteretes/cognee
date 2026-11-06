"""Deprecated ``code_graph_pipeline`` alias on the dataset status endpoints (SDK-793).

Code repositories are built under ``cognify_pipeline`` now, so legacy pollers that
still ask for ``code_graph_pipeline`` see cognify's state, keyed as they asked, with
one deprecation warning per process. The processing status counts an item completed
under either stamp: a pre-upgrade build left only the ``code_graph_pipeline`` one.
"""

import importlib
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
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


OLDER = datetime(2026, 1, 1, tzinfo=timezone.utc)
NEWER = OLDER + timedelta(days=30)


def _fake_runs(monkeypatch, runs):
    """``runs`` maps pipeline name -> {dataset_id: status | (status, created_at)};
    stands in for the latest-run lookups behind get_status / get_progress / the
    alias resolver. A bare status gets ``OLDER`` as its created_at."""
    datasets_module = importlib.import_module(DATASETS_MODULE)
    calls = []

    def _row(entry):
        status, created_at = entry if isinstance(entry, tuple) else (entry, OLDER)
        return SimpleNamespace(status=PipelineRunStatus(status), created_at=created_at)

    async def fake_latest_runs(dataset_ids, pipeline_name):
        by_dataset = runs.get(pipeline_name, {})
        return {d: _row(by_dataset[d]) for d in dataset_ids if d in by_dataset}

    async def fake_status(dataset_ids, pipeline_name):
        calls.append((list(dataset_ids), pipeline_name))
        latest = await fake_latest_runs(dataset_ids, pipeline_name)
        return {str(d): run.status.value for d, run in latest.items()}

    async def fake_progress(dataset_ids, pipeline_name):
        statuses = await fake_status(dataset_ids, pipeline_name)
        return {d: {"status": s, "progress": None} for d, s in statuses.items()}

    monkeypatch.setattr(datasets_module, "get_pipeline_status", fake_status)
    monkeypatch.setattr(datasets_module, "get_pipeline_progress", fake_progress)
    return calls


def test_alias_reads_cognify_pipeline(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {"cognify_pipeline": {dataset_id: "DATASET_PROCESSING_COMPLETED"}})

    response = client.get(
        "/api/v1/datasets/status",
        params={"dataset": str(dataset_id), "pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    assert response.json() == {str(dataset_id): "DATASET_PROCESSING_COMPLETED"}


def test_alias_deprecation_is_logged_once_per_process(client, monkeypatch):
    datasets_module = importlib.import_module(DATASETS_MODULE)
    monkeypatch.setattr(datasets_module, "_legacy_alias_warned", False)
    warning = Mock()
    monkeypatch.setattr(datasets_module.logger, "warning", warning)
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(monkeypatch, {"cognify_pipeline": {dataset_id: "DATASET_PROCESSING_STARTED"}})

    for _ in range(3):
        response = client.get(
            "/api/v1/datasets/status",
            params={"dataset": str(dataset_id), "pipeline": "code_graph_pipeline"},
        )
        assert response.status_code == 200

    assert warning.call_count == 1


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


def test_processing_status_alias_counts_either_stamp(client, monkeypatch):
    """Pre-upgrade items carry only the code_graph_pipeline stamp, post-upgrade
    items only the cognify_pipeline one: both count as completed, from one read."""
    dataset_id = uuid.uuid4()
    key = str(dataset_id)
    done = "DATA_ITEM_PROCESSING_COMPLETED"
    rows = [
        SimpleNamespace(id=uuid.uuid4(), name="old", pipeline_status=stamps)
        for stamps in (
            {"code_graph_pipeline": {key: done}},
            {"cognify_pipeline": {key: done}},
            {"add_pipeline": {key: done}},
        )
    ]
    _authorize(monkeypatch, [dataset_id])
    helper_module = importlib.import_module(
        "cognee.modules.data.methods.get_dataset_processing_status"
    )
    read_rows = AsyncMock(return_value=rows)
    monkeypatch.setattr(helper_module, "get_dataset_data", read_rows)

    response = client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["completed"], body["pending"]) == (3, 2, 1)
    assert [item["completed"] for item in body["items"]] == [True, True, False]
    read_rows.assert_awaited_once()
