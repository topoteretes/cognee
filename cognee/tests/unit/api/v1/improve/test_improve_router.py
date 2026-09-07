"""POST /v1/improve: every improve() option is reachable and the reply is the ImproveResult.

The orchestrator is stubbed; these tests pin the HTTP surface (plan A4): the
payload fields the router forwards, the defaults it applies, and the shape it
serializes back — one entry per stage, on ``PipelineRunInfo``'s statuses.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.improve.routers.get_improve_router import get_improve_router
from cognee.modules.improve import ImproveResult, StageResult
from cognee.modules.pipelines.models import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())
DATASET_ID = uuid4()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_improve_router(), prefix="/improve")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


@pytest.fixture
def improve_stub(monkeypatch):
    """Replace the orchestrator with a mock returning a canned ImproveResult."""
    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    run_info = PipelineRunCompleted(
        pipeline_run_id=uuid4(), dataset_id=DATASET_ID, dataset_name="docs", status="completed"
    )
    enrichment = StageResult.from_pipeline_run("triplet_enrichment", {DATASET_ID: run_info})
    result = ImproveResult(
        dataset_id=DATASET_ID,
        dataset_name="docs",
        session_ids=["s1"],
        stages=[
            StageResult.skipped("feedback_weights", "feedback_influence_zero"),
            StageResult.completed("persist_session_qa", entries=3),
            enrichment,
        ],
        memify_run={DATASET_ID: run_info},
    )
    stub = AsyncMock(return_value=result)
    monkeypatch.setattr(improve_pkg, "improve", stub)
    return stub


def test_requires_a_dataset(client, improve_stub):
    resp = client.post("/improve", json={})

    assert resp.status_code == 400
    improve_stub.assert_not_called()


def test_response_is_the_improve_result_with_one_entry_per_stage(client, improve_stub):
    resp = client.post("/improve", json={"datasetName": "docs", "sessionIds": ["s1"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["dataset_id"] == str(DATASET_ID)
    assert body["session_ids"] == ["s1"]
    assert [stage["stage"] for stage in body["stages"]] == [
        "feedback_weights",
        "persist_session_qa",
        "triplet_enrichment",
    ]
    assert body["stages"][0] == {
        "stage": "feedback_weights",
        "status": "skipped",
        "reason": "feedback_influence_zero",
        "error": None,
        "counts": {},
        "llm_calls": 0,
        "duration_ms": 0,
        "run": None,
    }
    assert body["stages"][1]["counts"] == {"entries": 3}
    # A pipeline-backed stage carries its run info, and the legacy memify
    # return stays reachable as memify_run (D4) — UUID keys serialize as strings.
    assert body["stages"][2]["run"]["status"] == "completed"
    assert body["memify_run"][str(DATASET_ID)]["status"] == "completed"


def test_forwards_every_option_to_improve(client, improve_stub):
    resp = client.post(
        "/improve",
        json={
            "datasetId": str(DATASET_ID),
            "sessionIds": ["s1", "s2"],
            "nodeName": ["Alice"],
            "buildGlobalContextIndex": True,
            "buildTruthSubspace": True,
            "feedbackAlpha": 0.25,
            "runInBackground": True,
        },
    )

    assert resp.status_code == 200
    kwargs = improve_stub.call_args.kwargs
    assert kwargs["dataset"] == DATASET_ID
    assert kwargs["session_ids"] == ["s1", "s2"]
    assert kwargs["node_name"] == ["Alice"]
    assert kwargs["build_global_context_index"] is True
    assert kwargs["build_truth_subspace"] is True
    assert kwargs["feedback_alpha"] == 0.25
    assert kwargs["run_in_background"] is True
    assert kwargs["user"] is MOCK_USER


def test_defaults_leave_feedback_alpha_to_the_server_config(client, improve_stub):
    """No feedback_alpha in the body means the kwarg is not passed, so
    IMPROVE_FEEDBACK_ALPHA applies; data defaults to None like the memify DTO."""
    resp = client.post("/improve", json={"datasetName": "docs"})

    assert resp.status_code == 200
    kwargs = improve_stub.call_args.kwargs
    assert "feedback_alpha" not in kwargs
    assert kwargs["data"] is None
    assert kwargs["build_global_context_index"] is False
    assert kwargs["build_truth_subspace"] is False
    assert kwargs["run_in_background"] is False
    assert kwargs["session_ids"] is None


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_rejects_feedback_alpha_outside_unit_interval(client, improve_stub, alpha):
    resp = client.post("/improve", json={"datasetName": "docs", "feedbackAlpha": alpha})

    assert resp.status_code == 422
    improve_stub.assert_not_called()


def test_non_fatal_stage_error_is_reported_in_the_body(client, monkeypatch):
    """A stage that errored (fail-open) is visible per stage with a 200; only a
    raised (fatal) error becomes a 4xx."""
    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    result = ImproveResult(
        dataset_id=DATASET_ID,
        dataset_name="docs",
        stages=[
            StageResult.errored("persist_agent_traces", RuntimeError("boom")),
            StageResult.completed("triplet_enrichment"),
        ],
        memify_run={},
    )
    monkeypatch.setattr(improve_pkg, "improve", AsyncMock(return_value=result))

    resp = client.post("/improve", json={"datasetName": "docs"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "errored"
    assert body["stages"][0]["status"] == "errored"
    assert "boom" in body["stages"][0]["error"]


def test_background_run_reports_running_with_no_stages(client, monkeypatch):
    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    result = ImproveResult(dataset_id=DATASET_ID, dataset_name="docs", background=True)
    result.finished = False
    monkeypatch.setattr(improve_pkg, "improve", AsyncMock(return_value=result))

    resp = client.post("/improve", json={"datasetName": "docs", "runInBackground": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["background"] is True
    assert body["stages"] == []


def test_lost_lock_is_every_stage_skipped(client, monkeypatch):
    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    result = ImproveResult.all_skipped(
        ["feedback_weights", "triplet_enrichment"], "lock_held", dataset_name="docs"
    )
    monkeypatch.setattr(improve_pkg, "improve", AsyncMock(return_value=result))

    resp = client.post("/improve", json={"datasetName": "docs"})

    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "skipped"
    assert {stage["reason"] for stage in body["stages"]} == {"lock_held"}


def test_unexpected_error_is_a_generic_409(client, monkeypatch):
    improve_pkg = importlib.import_module("cognee.api.v1.improve")
    monkeypatch.setattr(improve_pkg, "improve", AsyncMock(side_effect=RuntimeError("secret")))

    resp = client.post("/improve", json={"datasetName": "docs"})

    assert resp.status_code == 409
    assert "secret" not in resp.text


def test_docstring_no_longer_promises_a_graph_to_session_sync():
    router = get_improve_router()
    route = next(r for r in router.routes if getattr(r, "path", None) == "")
    doc = (route.endpoint.__doc__ or "").lower()
    assert "sync" not in doc
    assert "build_truth_subspace" in doc
    assert "feedback_alpha" in doc
