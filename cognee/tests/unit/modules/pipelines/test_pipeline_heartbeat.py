"""Tests for the pipeline-run progress heartbeat.

The heartbeat exists to answer one question: is a long-running pipeline still
alive, or abandoned? These tests cover the two properties that make the answer
trustworthy: the stamp lands on the right row, and producing it stays cheap
enough to run at every task boundary.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations import heartbeat_pipeline_run as heartbeat_module
from cognee.modules.pipelines.operations.heartbeat_pipeline_run import (
    heartbeat_pipeline_run,
    reset_heartbeat_throttle,
)


@pytest.fixture(autouse=True)
def _clean_throttle_state():
    reset_heartbeat_throttle()
    yield
    reset_heartbeat_throttle()


@pytest.fixture
def write_every_time(monkeypatch):
    """Disable throttling so each call attempts a write."""
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "0")


# --------------------------------------------------------------------------
# Fakes for the throttling / robustness tests, which never need a real DB.
# --------------------------------------------------------------------------


class _RecordingSession:
    def __init__(self, recorder, delay=0.0, error=None):
        self._recorder = recorder
        self._delay = delay
        self._error = error

    async def execute(self, statement):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        self._recorder.append(statement)

    async def commit(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _RecordingEngine:
    def __init__(self, delay=0.0, error=None):
        self.statements = []
        self._delay = delay
        self._error = error

    def get_async_session(self):
        return _RecordingSession(self.statements, self._delay, self._error)


@pytest.mark.asyncio
async def test_heartbeat_is_a_no_op_without_a_run_id(monkeypatch):
    engine = _RecordingEngine()
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)

    assert await heartbeat_pipeline_run(None) is False
    assert engine.statements == []


@pytest.mark.asyncio
async def test_heartbeat_writes_then_throttles_subsequent_calls(monkeypatch):
    """Progress triggers writes; the throttle caps how many reach the database.

    handle_task fires once per task per streamed batch, so an unthrottled
    heartbeat would mean thousands of UPDATEs against one row per ingestion.
    """
    engine = _RecordingEngine()
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "30")

    pipeline_run_id = uuid4()

    assert await heartbeat_pipeline_run(pipeline_run_id) is True
    for _ in range(50):
        assert await heartbeat_pipeline_run(pipeline_run_id) is False

    assert len(engine.statements) == 1


@pytest.mark.asyncio
async def test_throttle_is_per_run(monkeypatch):
    """One run's heartbeat must not suppress another concurrent run's."""
    engine = _RecordingEngine()
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "30")

    assert await heartbeat_pipeline_run(uuid4()) is True
    assert await heartbeat_pipeline_run(uuid4()) is True

    assert len(engine.statements) == 2


@pytest.mark.asyncio
async def test_concurrent_task_completions_produce_one_write(monkeypatch):
    """run_tasks gathers up to data_per_batch items per run, so many task
    completions land in the same throttle window at once. The write slot is
    claimed before awaiting the database, so only one of them writes."""
    engine = _RecordingEngine(delay=0.01)
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "30")

    pipeline_run_id = uuid4()
    results = await asyncio.gather(*[heartbeat_pipeline_run(pipeline_run_id) for _ in range(20)])

    assert results.count(True) == 1
    assert len(engine.statements) == 1


@pytest.mark.asyncio
async def test_heartbeat_failure_never_propagates(monkeypatch, write_every_time):
    """A heartbeat is an observability signal, so losing one must not fail the
    pipeline task that produced it."""
    engine = _RecordingEngine(error=RuntimeError("database is gone"))
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)

    assert await heartbeat_pipeline_run(uuid4()) is False


@pytest.mark.asyncio
async def test_negative_interval_disables_heartbeats(monkeypatch):
    engine = _RecordingEngine()
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "-1")

    assert await heartbeat_pipeline_run(uuid4()) is False
    assert engine.statements == []


@pytest.mark.asyncio
async def test_invalid_interval_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "not-a-number")

    assert (
        heartbeat_module.get_heartbeat_interval_seconds()
        == heartbeat_module.DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    )


# --------------------------------------------------------------------------
# Wiring: a running pipeline must actually emit the signal.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_a_pipeline_heartbeats_each_completed_task(monkeypatch):
    from types import SimpleNamespace

    from cognee.modules.pipelines.models import PipelineContext
    from cognee.modules.pipelines.operations import run_tasks_base as run_tasks_base_module
    from cognee.modules.pipelines.tasks.task import Task

    beats = []

    async def _record(pipeline_run_id):
        beats.append(pipeline_run_id)
        return True

    monkeypatch.setattr(run_tasks_base_module, "heartbeat_pipeline_run", _record)

    async def emit(count):
        for value in range(count):
            yield value

    async def double(values):
        yield [value * 2 for value in values]

    pipeline_run_id = uuid4()
    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=None, email=None),
        pipeline_run_id=pipeline_run_id,
        pipeline_name="heartbeat_test_pipeline",
    )

    pipeline = run_tasks_base_module.run_tasks_base(
        [Task(emit), Task(double)],
        data=3,
        user=ctx.user,
        ctx=ctx,
    )
    async for _ in pipeline:
        pass

    assert beats, "a completed task must report progress"
    assert set(beats) == {pipeline_run_id}


@pytest.mark.asyncio
async def test_pipeline_without_a_run_id_does_not_heartbeat(monkeypatch):
    """run_pipeline builds a PipelineContext with no pipeline_run_id because it
    writes no PipelineRun row, so there is nothing to stamp."""
    from types import SimpleNamespace

    from cognee.modules.pipelines.models import PipelineContext
    from cognee.modules.pipelines.operations import run_tasks_base as run_tasks_base_module
    from cognee.modules.pipelines.tasks.task import Task

    engine = _RecordingEngine()
    monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)
    monkeypatch.setenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS", "0")

    async def emit(count):
        for value in range(count):
            yield value

    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=None, email=None),
        pipeline_name="no_run_id_pipeline",
    )

    async for _ in run_tasks_base_module.run_tasks_base(
        [Task(emit)], data=2, user=ctx.user, ctx=ctx
    ):
        pass

    assert engine.statements == []


# --------------------------------------------------------------------------
# Round trip against a real SQLite database: proves the UPDATE targets the
# right row of the append-only PipelineRun log.
# --------------------------------------------------------------------------


@asynccontextmanager
async def sqlite_engine(tmp_path):
    """A throwaway SQLite database with the real schema.

    Written as a context manager rather than a pytest fixture because
    pytest-asyncio runs in strict mode here, where plain ``@pytest.fixture``
    async generators are not awaited.
    """
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'heartbeat_test.db'}")
    await adapter.create_database()
    try:
        yield adapter
    finally:
        await adapter.engine.dispose()


async def _insert(engine, rows):
    async with engine.get_async_session() as session:
        for row in rows:
            session.add(row)
        await session.commit()


async def _fetch(engine, pipeline_run_id):
    async with engine.get_async_session() as session:
        result = await session.execute(
            select(PipelineRun).where(PipelineRun.pipeline_run_id == pipeline_run_id)
        )
        return result.scalars().all()


@pytest.mark.asyncio
async def test_heartbeat_stamps_only_the_started_row(monkeypatch, tmp_path, write_every_time):
    async with sqlite_engine(tmp_path) as engine:
        monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)

        pipeline_run_id = uuid4()
        dataset_id = uuid4()
        started_at = datetime.now(timezone.utc) - timedelta(days=3)

        await _insert(
            engine,
            [
                PipelineRun(
                    pipeline_run_id=pipeline_run_id,
                    pipeline_name="cognify_pipeline",
                    pipeline_id=uuid4(),
                    dataset_id=dataset_id,
                    status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
                    created_at=started_at,
                    run_info={},
                ),
                # The INITIATED row of the same run must be left alone: only
                # the STARTED row represents "in flight".
                PipelineRun(
                    pipeline_run_id=pipeline_run_id,
                    pipeline_name="cognify_pipeline",
                    pipeline_id=uuid4(),
                    dataset_id=dataset_id,
                    status=PipelineRunStatus.DATASET_PROCESSING_INITIATED,
                    created_at=started_at,
                    run_info={},
                ),
            ],
        )

        assert await heartbeat_pipeline_run(pipeline_run_id) is True

        rows = {row.status: row for row in await _fetch(engine, pipeline_run_id)}

        started = rows[PipelineRunStatus.DATASET_PROCESSING_STARTED]
        assert started.last_heartbeat_at is not None
        # The run started three days ago but just reported progress, exactly
        # the long-local-LLM-run case that age alone cannot tell from a stall.
        assert started.last_heartbeat_at.replace(tzinfo=timezone.utc) > started_at

        assert rows[PipelineRunStatus.DATASET_PROCESSING_INITIATED].last_heartbeat_at is None


@pytest.mark.asyncio
async def test_heartbeat_does_not_touch_other_runs(monkeypatch, tmp_path, write_every_time):
    async with sqlite_engine(tmp_path) as engine:
        monkeypatch.setattr(heartbeat_module, "get_relational_engine", lambda: engine)

        beating_run_id = uuid4()
        other_run_id = uuid4()

        await _insert(
            engine,
            [
                PipelineRun(
                    pipeline_run_id=run_id,
                    pipeline_name="cognify_pipeline",
                    pipeline_id=uuid4(),
                    dataset_id=uuid4(),
                    status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
                    created_at=datetime.now(timezone.utc),
                    run_info={},
                )
                for run_id in (beating_run_id, other_run_id)
            ],
        )

        await heartbeat_pipeline_run(beating_run_id)

        assert (await _fetch(engine, beating_run_id))[0].last_heartbeat_at is not None
        assert (await _fetch(engine, other_run_id))[0].last_heartbeat_at is None
