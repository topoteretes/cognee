from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import recovery as recovery_module
from cognee.modules.pipelines.models import PipelineRunStatus


class _FakeScalarsResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _FakeExecuteResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _FakeScalarsResult(self._items)


class _FakeSession:
    def __init__(self, execute_result=None, dataset=None):
        self._execute_result = execute_result
        self._dataset = dataset

    async def execute(self, _statement):
        return self._execute_result

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

    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=pipeline_run_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    discovery_session = _FakeSession(execute_result=_FakeExecuteResult([stale_run]))
    dataset_session = _FakeSession(dataset=dataset)
    engine = _FakeEngine([discovery_session, dataset_session])

    rollback_calls = []
    reset_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _reset_status(**kwargs):
        reset_calls.append(kwargs)

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "reset_pipeline_run_status", _reset_status)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert rollback_calls[0]["pipeline_run_id"] == pipeline_run_id
    assert rollback_calls[0]["dataset"] == dataset
    # The lingering STARTED status must be reset so a re-run is not blocked.
    assert len(reset_calls) == 1
    assert reset_calls[0]["dataset_id"] == dataset_id
    assert reset_calls[0]["pipeline_name"] == "cognify_pipeline"


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

    discovery_session = _FakeSession(execute_result=_FakeExecuteResult([stale_run]))
    dataset_session = _FakeSession(dataset=None)
    engine = _FakeEngine([discovery_session, dataset_session])

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_swallows_discovery_query_failure(monkeypatch):
    """A failure while querying for stale runs must not propagate: the
    function is called from FastAPI lifespan, so an exception here would
    prevent the whole API from starting."""

    class _FailingSession(_FakeSession):
        async def execute(self, _statement):
            raise RuntimeError("relational database unavailable")

    engine = _FakeEngine([_FailingSession()])

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
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

    discovery_session = _FakeSession(execute_result=_FakeExecuteResult([recent_run]))
    # No dataset session is consumed because the run is skipped before lookup.
    engine = _FakeEngine([discovery_session])

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "STALE_RUN_MIN_AGE_SECONDS", 3600)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
