"""Pipeline rows inherit the sessions of the operation that launched them.

An ``improve`` given sessions stamps them on its own operation record, but the
dashboard lists the ``memify_pipeline`` rows the run wrote. Without inheritance
those carry no session, so every session page shows every improve run on the
dataset (CLO-1105). A row's sessions live in ``pipeline_run_sessions`` only, one
row per session.

Runs against a real temporary SQLite database — no LLM, no network.
"""

import asyncio
import importlib
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.operations import (
    get_current_session_ids,
    parent_run_scope,
    record_operation,
)
from cognee.modules.pipelines.models.PipelineRun import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.models.PipelineRunSession import PipelineRunSession

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")
start_mod = importlib.import_module("cognee.modules.pipelines.operations.log_pipeline_run_start")
complete_mod = importlib.import_module(
    "cognee.modules.pipelines.operations.log_pipeline_run_complete"
)
error_mod = importlib.import_module("cognee.modules.pipelines.operations.log_pipeline_run_error")
run_sessions_mod = importlib.import_module("cognee.modules.pipelines.operations.run_sessions")
recovery_mod = importlib.import_module("cognee.modules.cognify.recovery")


@pytest_asyncio.fixture
async def ops_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs and pipeline_run_sessions tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="pipeline_session_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[PipelineRun.__table__, PipelineRunSession.__table__],
        )

    for module in (record_operation_mod, start_mod, complete_mod, error_mod, run_sessions_mod):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _rows(engine):
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRun))
        return result.scalars().all()


async def _sessions(engine) -> dict:
    """pipeline_runs row id -> its sessions, sorted."""
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRunSession))
        sessions = {}
        for row in result.scalars().all():
            sessions.setdefault(row.pipeline_run_row_id, []).append(row.session_id)
    return {row_id: sorted(ids) for row_id, ids in sessions.items()}


async def _run_memify(dataset_id):
    """The started + completed rows run_tasks writes for one pipeline run."""
    started = await start_mod.log_pipeline_run_start(uuid4(), "memify_pipeline", dataset_id)
    with parent_run_scope(started.pipeline_run_id):
        await complete_mod.log_pipeline_run_complete(
            started.pipeline_run_id, started.pipeline_id, "memify_pipeline", dataset_id
        )
    return started.pipeline_run_id


@pytest.mark.asyncio
async def test_pipeline_rows_carry_the_operations_session_id(ops_engine):
    """improve binds the session after the scope opens; the stamp is read at write time."""
    async with record_operation("improve") as operation:
        operation.set_session_ids(["chat_1"])
        run_id = await _run_memify(uuid4())

    rows = await _rows(ops_engine)
    memify_rows = [row for row in rows if row.pipeline_run_id == run_id]
    assert {row.status for row in memify_rows} == {
        PipelineRunStatus.DATASET_PROCESSING_STARTED,
        PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
    }
    sessions = await _sessions(ops_engine)
    assert all(sessions[row.id] == ["chat_1"] for row in memify_rows)
    # The operation record keeps its own stamp.
    assert all(sessions[row.id] == ["chat_1"] for row in rows)


@pytest.mark.asyncio
async def test_errored_row_carries_the_session_id(ops_engine):
    run_id = uuid4()
    async with record_operation("improve") as operation:
        operation.set_session_ids(["chat_1"])
        await error_mod.log_pipeline_run_error(
            run_id, uuid4(), "memify_pipeline", uuid4(), ValueError("boom")
        )

    rows = [row for row in await _rows(ops_engine) if row.pipeline_run_id == run_id]
    assert len(rows) == 1
    assert (await _sessions(ops_engine))[rows[0].id] == ["chat_1"]


@pytest.mark.asyncio
async def test_explicit_session_ids_win_for_recovery(ops_engine):
    """Startup recovery closes a dead run from a context with no operation: it
    passes the STARTED row's sessions, which must not be replaced."""
    run_id = uuid4()
    await error_mod.log_pipeline_run_error(
        run_id,
        uuid4(),
        "memify_pipeline",
        uuid4(),
        ValueError("gone"),
        session_ids=("chat_1",),
    )

    rows = await _rows(ops_engine)
    assert len(rows) == 1
    assert (await _sessions(ops_engine))[rows[0].id] == ["chat_1"]


@pytest.mark.asyncio
async def test_no_session_outside_a_session_bound_operation(ops_engine):
    assert get_current_session_ids() == ()
    await _run_memify(uuid4())

    async with record_operation("improve"):
        await _run_memify(uuid4())

    assert len(await _rows(ops_engine)) == 5  # two runs of two rows, one operation record
    assert await _sessions(ops_engine) == {}


@pytest.mark.asyncio
async def test_background_run_keeps_the_session_after_the_scope_exits(ops_engine):
    """improve(run_in_background=True) launches the stages as a task and leaves
    the scope; the task's rows are written after it, from its copied context."""
    release = asyncio.Event()
    run_ids = []

    async def stages():
        await release.wait()
        run_ids.append(await _run_memify(uuid4()))

    async with record_operation("improve") as operation:
        operation.set_session_ids(["chat_1"])
        operation.defer_close()
        task = asyncio.create_task(stages())

    assert get_current_session_ids() == ()
    release.set()
    await task

    rows = [row for row in await _rows(ops_engine) if row.pipeline_run_id == run_ids[0]]
    assert len(rows) == 2
    sessions = await _sessions(ops_engine)
    assert all(sessions[row.id] == ["chat_1"] for row in rows)


@pytest.mark.asyncio
async def test_a_multi_session_operation_stamps_every_session_on_every_row(ops_engine):
    """improve(session_ids=[a, b]): no single session_id, both sessions listed per row."""
    async with record_operation("improve") as operation:
        operation.set_session_ids(["chat_1", "chat_2"])
        run_id = await _run_memify(uuid4())

    rows = await _rows(ops_engine)
    assert len(rows) == 3  # STARTED, COMPLETED, the operation record
    sessions = await _sessions(ops_engine)
    assert all(sessions[row.id] == ["chat_1", "chat_2"] for row in rows)
    assert {row.pipeline_run_id for row in rows if row.pipeline_name} == {run_id}


@pytest.mark.asyncio
async def test_session_ids_keep_order_and_drop_duplicates_and_blanks(ops_engine):
    async with record_operation("improve") as operation:
        operation.set_session_ids(["chat_2", "", "chat_1", "chat_2", None])
        assert operation.session_ids == ("chat_2", "chat_1")
        operation.set_session_ids(["chat_1"])
        assert operation.session_ids == ("chat_1",)


@pytest.mark.asyncio
async def test_recovery_carries_every_session_of_the_started_row(ops_engine):
    """A multi-session run that died is closed with all of its STARTED row's sessions."""
    async with record_operation("improve") as operation:
        operation.set_session_ids(["chat_1", "chat_2"])
        started = await start_mod.log_pipeline_run_start(uuid4(), "memify_pipeline", uuid4())

    class _Dataset:
        id = started.dataset_id

    # Recovery reads every candidate's sessions in one batch and hands each
    # run its own; the closer itself runs no query.
    session_ids = (await _sessions(ops_engine))[started.id]
    await recovery_mod._close_as_abandoned(started, _Dataset(), session_ids)

    errored = [
        row
        for row in await _rows(ops_engine)
        if row.status == PipelineRunStatus.DATASET_PROCESSING_ERRORED
    ]
    assert len(errored) == 1
    assert (await _sessions(ops_engine))[errored[0].id] == ["chat_1", "chat_2"]
