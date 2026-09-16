from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import recovery as recovery_module
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.models import PipelineRunStatus


class _FakeSession:
    def __init__(self, dataset=None):
        self._dataset = dataset

    async def get(self, _model, _dataset_id):
        return self._dataset

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get_async_session(self):
        return self._sessions.pop(0)


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_executes_rollback_for_latest_candidate(monkeypatch):
    dataset_id = uuid4()
    owner_id = uuid4()
    pipeline_run_id = uuid4()

    pipeline_id = uuid4()
    started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        pipeline_run_id=pipeline_run_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=started_at,
        started_at=started_at,
        user_id=owner_id,
        tenant_id=None,
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    dataset_session = _FakeSession(dataset=dataset)
    engine = _FakeEngine([dataset_session])

    calls = []

    async def _rollback_handler(**kwargs):
        calls.append(("rollback", kwargs))

    async def _log_error(**kwargs):
        calls.append(("error", kwargs))

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    # Rollback first, then the run is closed as ERRORED; never the other way round.
    assert [name for name, _ in calls] == ["rollback", "error"]
    rollback = calls[0][1]
    assert rollback["pipeline_run_id"] == pipeline_run_id
    assert rollback["dataset"] == dataset
    error = calls[1][1]
    assert error["pipeline_run_id"] == pipeline_run_id
    assert error["pipeline_id"] == pipeline_id
    assert error["pipeline_name"] == "cognify_pipeline"
    assert error["dataset_id"] == dataset_id
    assert isinstance(error["e"], AbandonedPipelineRunError)
    assert error["user"].id == owner_id
    assert error["started_at"] == started_at


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_leaves_run_started_when_rollback_fails(monkeypatch):
    """A failed rollback must not close the run: leaving it STARTED makes the next boot retry."""
    dataset_id = uuid4()
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        pipeline_id=uuid4(),
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=None,
        user_id=None,
        tenant_id=None,
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    engine = _FakeEngine([_FakeSession(dataset=dataset)])

    error_calls = []

    async def _rollback_handler(**_kwargs):
        raise RuntimeError("graph store unavailable")

    async def _log_error(**kwargs):
        error_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert error_calls == []


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_skips_missing_dataset(monkeypatch):
    dataset_id = uuid4()
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )

    dataset_session = _FakeSession(dataset=None)
    engine = _FakeEngine([dataset_session])

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_skips_recent_run(monkeypatch):
    """A STARTED run younger than the staleness threshold is left alone so a
    live run on another worker is not rolled back out from under it."""
    dataset_id = uuid4()
    recent_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc),
    )

    # No dataset session is consumed because the run is skipped before lookup.
    engine = _FakeEngine([])

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: recent_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "STALE_RUN_MIN_AGE_SECONDS", 3600)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
