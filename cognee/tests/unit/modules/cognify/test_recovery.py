from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import recovery as recovery_module
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
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=pipeline_run_id,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        run_info={"data": "summarized-payload"},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    dataset_session = _FakeSession(dataset=dataset)
    engine = _FakeEngine([dataset_session])

    rollback_calls = []
    close_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert rollback_calls[0]["pipeline_run_id"] == pipeline_run_id
    assert rollback_calls[0]["dataset"] == dataset
    # The run is closed as ERRORED, keeping its identity, and the error class
    # is what distinguishes a killed run from one whose work actually failed.
    assert len(close_calls) == 1
    assert close_calls[0]["dataset_id"] == dataset_id
    assert close_calls[0]["pipeline_run_id"] == pipeline_run_id
    assert close_calls[0]["pipeline_id"] == pipeline_id
    assert type(close_calls[0]["e"]).__name__ == "AbandonedPipelineRunError"
    # The STARTED row's already-summarized payload is passed through rather
    # than re-summarized.
    assert close_calls[0]["data_info"] == "summarized-payload"


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
async def test_a_recent_run_is_recovered_too(monkeypatch):
    """Age is not the signal. A pipeline executes inside the API process, so a
    STARTED row found while that process is starting belonged to a process that
    is gone, however recently it began. The old threshold skipped these, which
    left a run that died moments before a restart reported as processing until
    some later boot."""
    dataset_id = uuid4()
    owner_id = uuid4()
    recent_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        pipeline_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        run_info={},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    engine = _FakeEngine([_FakeSession(dataset=dataset)])

    rollback_calls = []
    close_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: recent_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_a_failing_rollback_leaves_the_run_open_for_the_next_boot(monkeypatch):
    """The STARTED row is the retry token. Closing the run before the rollback
    finished would mark it terminal over a half-deleted graph that nothing
    would ever revisit, so a raising rollback must leave it open instead."""
    dataset_id = uuid4()
    owner_id = uuid4()
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        pipeline_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        run_info={},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    engine = _FakeEngine([_FakeSession(dataset=dataset)])

    close_calls = []

    async def _rollback_handler(**kwargs):
        raise RuntimeError("graph store unreachable")

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    # Startup must survive it: one dataset failing to recover cannot stop the
    # server from coming up.
    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert close_calls == []
