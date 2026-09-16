"""Startup recovery: every pipeline's STARTED runs are rolled back (if the pipeline has a
rollback) and closed as ERRORED with the STARTED row's own metadata; nothing else is touched."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import recovery as recovery_module
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.models import PipelineRunStatus

TWO_HOURS_AGO = datetime.now(timezone.utc) - timedelta(hours=2)


class _FakeSession:
    def __init__(self, datasets):
        self._datasets = datasets

    async def get(self, _model, dataset_id):
        return self._datasets.get(dataset_id)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, datasets):
        self._datasets = datasets

    def get_async_session(self):
        return _FakeSession(self._datasets)


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


def _run(pipeline_name, status=PipelineRunStatus.DATASET_PROCESSING_STARTED, **overrides):
    fields = {
        "pipeline_name": pipeline_name,
        "pipeline_id": uuid4(),
        "dataset_id": uuid4(),
        "pipeline_run_id": uuid4(),
        "status": status,
        "created_at": TWO_HOURS_AGO,
        "started_at": TWO_HOURS_AGO,
        "user_id": uuid4(),
        "tenant_id": uuid4(),
        "run_info": {"data": ["doc-1", "doc-2"]},
        "origin": "api",
        "parent_operation_id": uuid4(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _wire(monkeypatch, runs, datasets, *, rollback_fails=False):
    """Patch the module's collaborators and return the ordered list of calls made."""
    calls = []

    async def _rollback(**kwargs):
        if rollback_fails:
            raise RuntimeError("graph store unavailable")
        calls.append(("rollback", kwargs))

    async def _log_error(**kwargs):
        calls.append(("error", kwargs))

    async def _latest_runs():
        return runs

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: _FakeEngine(datasets))
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_for_all_pipelines", _latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "ROLLBACK_HANDLERS", {"cognify_pipeline": _rollback})
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)
    monkeypatch.setattr(recovery_module, "STALE_RUN_MIN_AGE_SECONDS", 3600)
    return calls


def _dataset_for(run):
    return SimpleNamespace(id=run.dataset_id, owner_id=uuid4())


@pytest.mark.asyncio
async def test_cognify_run_is_rolled_back_then_closed_with_its_own_metadata(monkeypatch):
    run = _run("cognify_pipeline")
    dataset = _dataset_for(run)
    calls = _wire(monkeypatch, [run], {run.dataset_id: dataset})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    # Rollback first, then the ERRORED row; never the other way round.
    assert [name for name, _ in calls] == ["rollback", "error"]
    rollback = calls[0][1]
    assert rollback["pipeline_run_id"] == run.pipeline_run_id
    assert rollback["dataset"] == dataset

    error = calls[1][1]
    assert isinstance(error["e"], AbandonedPipelineRunError)
    assert error["pipeline_run_id"] == run.pipeline_run_id
    assert error["pipeline_id"] == run.pipeline_id
    assert error["pipeline_name"] == "cognify_pipeline"
    assert error["dataset_id"] == run.dataset_id
    # Everything the STARTED row knew about the run is carried onto the ERRORED row.
    assert error["user"].id == run.user_id
    assert error["user"].tenant_id == run.tenant_id
    assert error["started_at"] == run.started_at
    assert error["data_info"] == ["doc-1", "doc-2"]
    assert error["origin"] == "api"
    assert error["parent_operation_id"] == run.parent_operation_id


@pytest.mark.asyncio
async def test_pipeline_without_rollback_is_only_closed(monkeypatch):
    run = _run("add_pipeline")
    calls = _wire(monkeypatch, [run], {run.dataset_id: _dataset_for(run)})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert [name for name, _ in calls] == ["error"]
    error = calls[0][1]
    assert error["pipeline_name"] == "add_pipeline"
    assert isinstance(error["e"], AbandonedPipelineRunError)


@pytest.mark.asyncio
async def test_every_pipeline_with_a_stale_started_run_is_closed(monkeypatch):
    runs = [_run("cognify_pipeline"), _run("add_pipeline"), _run("memify_pipeline")]
    calls = _wire(monkeypatch, runs, {r.dataset_id: _dataset_for(r) for r in runs})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = sorted(kw["pipeline_name"] for name, kw in calls if name == "error")
    assert closed == ["add_pipeline", "cognify_pipeline", "memify_pipeline"]
    assert sum(1 for name, _ in calls if name == "rollback") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        PipelineRunStatus.DATASET_PROCESSING_INITIATED,
    ],
)
async def test_run_not_left_started_is_untouched(monkeypatch, status):
    """An ERRORED run stays ERRORED (its inline rollback already ran); COMPLETED and
    INITIATED runs are not the recovery's business either."""
    run = _run("cognify_pipeline", status=status)
    calls = _wire(monkeypatch, [run], {run.dataset_id: _dataset_for(run)})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert calls == []


@pytest.mark.asyncio
async def test_user_less_run_is_closed_without_a_user(monkeypatch):
    run = _run("add_pipeline", user_id=None, tenant_id=None, origin=None, parent_operation_id=None)
    calls = _wire(monkeypatch, [run], {run.dataset_id: _dataset_for(run)})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    error = calls[0][1]
    assert error["user"] is None
    assert error["origin"] is None
    assert error["parent_operation_id"] is None


@pytest.mark.asyncio
async def test_failed_rollback_leaves_run_started(monkeypatch):
    """No ERRORED row over graph data that is still there; the next boot retries."""
    run = _run("cognify_pipeline")
    calls = _wire(monkeypatch, [run], {run.dataset_id: _dataset_for(run)}, rollback_fails=True)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert calls == []


@pytest.mark.asyncio
async def test_missing_dataset_is_skipped(monkeypatch):
    run = _run("cognify_pipeline")
    calls = _wire(monkeypatch, [run], {})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert calls == []


@pytest.mark.asyncio
async def test_recent_run_is_treated_as_live(monkeypatch):
    """A STARTED row younger than the floor may belong to a sibling process mid-deploy."""
    run = _run("cognify_pipeline", created_at=datetime.now(timezone.utc) - timedelta(minutes=5))
    calls = _wire(monkeypatch, [run], {run.dataset_id: _dataset_for(run)})

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert calls == []


@pytest.mark.asyncio
async def test_one_failure_does_not_stop_the_others(monkeypatch):
    failing = _run("cognify_pipeline")
    healthy = _run("add_pipeline")
    datasets = {
        failing.dataset_id: _dataset_for(failing),
        healthy.dataset_id: _dataset_for(healthy),
    }
    calls = _wire(monkeypatch, [failing, healthy], datasets, rollback_fails=True)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert [kw["pipeline_name"] for name, kw in calls if name == "error"] == ["add_pipeline"]
