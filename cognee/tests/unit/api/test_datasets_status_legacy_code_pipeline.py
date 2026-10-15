"""Deprecated ``code_graph_pipeline`` alias on the dataset status endpoints (SDK-793).

Code repositories are built under ``cognify_pipeline`` now, so legacy pollers that
still ask for ``code_graph_pipeline`` (cognee-mcp's client) must see cognify's state,
keyed as they asked. The alias reports whichever of the dataset's latest
``code_graph_pipeline`` and ``cognify_pipeline`` runs is newer: a dataset whose last
pre-upgrade build left a ``code_graph_pipeline`` run must still see a newer
cognify build finish, while a custom pipeline that keeps running under the old
name keeps reporting its own (newer) run.
"""

import importlib
import uuid
from datetime import datetime, timedelta, timezone
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
        return SimpleNamespace(status=status, created_at=created_at)

    async def fake_latest_runs(dataset_ids, pipeline_name):
        by_dataset = runs.get(pipeline_name, {})
        return {d: _row(by_dataset[d]) for d in dataset_ids if d in by_dataset}

    async def fake_status(dataset_ids, pipeline_name):
        calls.append((list(dataset_ids), pipeline_name))
        latest = await fake_latest_runs(dataset_ids, pipeline_name)
        return {str(d): run.status for d, run in latest.items()}

    async def fake_progress(dataset_ids, pipeline_name):
        statuses = await fake_status(dataset_ids, pipeline_name)
        return {d: {"status": s, "progress": None} for d, s in statuses.items()}

    monkeypatch.setattr(datasets_module, "get_latest_pipeline_runs_by_datasets", fake_latest_runs)
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


def test_alias_reports_own_run_when_code_graph_run_is_newer(client, monkeypatch):
    with_run, without_run = uuid.uuid4(), uuid.uuid4()
    _authorize(monkeypatch, [with_run, without_run])
    _fake_runs(
        monkeypatch,
        {
            "code_graph_pipeline": {with_run: ("DATASET_PROCESSING_STARTED", NEWER)},
            "cognify_pipeline": {
                with_run: ("DATASET_PROCESSING_COMPLETED", OLDER),
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


def test_alias_reports_newer_cognify_run_over_stale_code_graph_run(client, monkeypatch):
    """A pre-upgrade code_graph_pipeline run must not mask a newer cognify build."""
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    _fake_runs(
        monkeypatch,
        {
            "code_graph_pipeline": {dataset_id: ("DATASET_PROCESSING_COMPLETED", OLDER)},
            "cognify_pipeline": {dataset_id: ("DATASET_PROCESSING_STARTED", NEWER)},
        },
    )
    warnings = _capture_warnings(monkeypatch)

    for endpoint, expected in (
        ("/api/v1/datasets/status", "DATASET_PROCESSING_STARTED"),
        (
            "/api/v1/datasets/status/progress",
            {"status": "DATASET_PROCESSING_STARTED", "progress": None},
        ),
    ):
        response = client.get(
            endpoint, params={"dataset": str(dataset_id), "pipeline": "code_graph_pipeline"}
        )
        assert response.status_code == 200
        assert response.json() == {str(dataset_id): expected}
    assert len(warnings) == 2


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


def _item(item_id, completed):
    return {"id": item_id, "name": f"item-{item_id}", "completed": completed}


def _patch_count_helper(monkeypatch, items_by_pipeline=None):
    """``items_by_pipeline`` maps pipeline name -> {item_id: completed}; the item
    order is the dataset's, shared across pipelines like the real helper's."""
    items_by_pipeline = items_by_pipeline or {}
    item_ids = list(dict.fromkeys(i for items in items_by_pipeline.values() for i in items))
    methods_module = importlib.import_module("cognee.modules.data.methods")

    async def fake_count(dataset_id, pipeline_name):
        stamped = items_by_pipeline.get(pipeline_name, {})
        items = [_item(i, stamped.get(i, False)) for i in item_ids]
        completed = sum(1 for item in items if item["completed"])
        return {
            "total": len(items),
            "completed": completed,
            "pending": len(items) - completed,
            "items": items,
        }

    helper = AsyncMock(side_effect=fake_count)
    monkeypatch.setattr(methods_module, "get_dataset_processing_status", helper)
    return helper


def test_processing_status_alias_counts_either_stamp(client, monkeypatch):
    """Pre-upgrade items carry only the code_graph_pipeline stamp, post-upgrade
    items only the cognify_pipeline one: both count as completed."""
    dataset_id = uuid.uuid4()
    old_item, new_item, pending_item = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    warnings = _capture_warnings(monkeypatch)
    helper = _patch_count_helper(
        monkeypatch,
        {
            "code_graph_pipeline": {old_item: True, new_item: False, pending_item: False},
            "cognify_pipeline": {old_item: False, new_item: True, pending_item: False},
        },
    )

    response = client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"pipeline": "code_graph_pipeline"},
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["completed"], body["pending"]) == (3, 2, 1)
    assert {item["id"]: item["completed"] for item in body["items"]} == {
        str(old_item): True,
        str(new_item): True,
        str(pending_item): False,
    }
    assert sorted(call.kwargs["pipeline_name"] for call in helper.await_args_list) == [
        "code_graph_pipeline",
        "cognify_pipeline",
    ]
    assert len(warnings) == 1


def test_processing_status_other_pipelines_count_only_their_own_stamp(client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize(monkeypatch, [dataset_id])
    helper = _patch_count_helper(monkeypatch)

    response = client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"pipeline": "cognify_pipeline"},
    )

    assert response.status_code == 200
    helper.assert_awaited_once_with(dataset_id, pipeline_name="cognify_pipeline")
