"""Pipeline rows inherit the session id of the operation that launched them.

An ``improve`` given one session stamps that session on its own operation
record, but the dashboard lists the ``memify_pipeline`` rows the run wrote.
Without inheritance those carry no session id, so every session page shows
every improve run on the dataset (CLO-1105).

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
    get_current_session_id,
    parent_run_scope,
    record_operation,
)
from cognee.modules.pipelines.models.PipelineRun import PipelineRun, PipelineRunStatus

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")
start_mod = importlib.import_module("cognee.modules.pipelines.operations.log_pipeline_run_start")
complete_mod = importlib.import_module(
    "cognee.modules.pipelines.operations.log_pipeline_run_complete"
)
error_mod = importlib.import_module("cognee.modules.pipelines.operations.log_pipeline_run_error")


@pytest_asyncio.fixture
async def ops_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs table."""
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
        await connection.run_sync(Base.metadata.create_all, tables=[PipelineRun.__table__])

    for module in (record_operation_mod, start_mod, complete_mod, error_mod):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _rows(engine):
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRun))
        return result.scalars().all()


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
        operation.set_session_id("chat_1")
        run_id = await _run_memify(uuid4())

    rows = await _rows(ops_engine)
    memify_rows = [row for row in rows if row.pipeline_run_id == run_id]
    assert {row.status for row in memify_rows} == {
        PipelineRunStatus.DATASET_PROCESSING_STARTED,
        PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
    }
    assert all(row.session_id == "chat_1" for row in memify_rows)
    # The operation record keeps its own stamp.
    assert {row.session_id for row in rows} == {"chat_1"}


@pytest.mark.asyncio
async def test_errored_row_carries_the_session_id(ops_engine):
    run_id = uuid4()
    async with record_operation("improve") as operation:
        operation.set_session_id("chat_1")
        await error_mod.log_pipeline_run_error(
            run_id, uuid4(), "memify_pipeline", uuid4(), ValueError("boom")
        )

    rows = [row for row in await _rows(ops_engine) if row.pipeline_run_id == run_id]
    assert [row.session_id for row in rows] == ["chat_1"]


@pytest.mark.asyncio
async def test_explicit_session_id_wins_for_recovery(ops_engine):
    """Startup recovery closes a dead run from a context with no operation: it
    passes the STARTED row's session id, which must not be replaced."""
    run_id = uuid4()
    await error_mod.log_pipeline_run_error(
        run_id, uuid4(), "memify_pipeline", uuid4(), ValueError("gone"), session_id="chat_1"
    )

    rows = await _rows(ops_engine)
    assert [row.session_id for row in rows] == ["chat_1"]


@pytest.mark.asyncio
async def test_no_session_outside_a_session_bound_operation(ops_engine):
    assert get_current_session_id() is None
    await _run_memify(uuid4())

    async with record_operation("improve"):
        await _run_memify(uuid4())

    assert {row.session_id for row in await _rows(ops_engine)} == {None}


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
        operation.set_session_id("chat_1")
        operation.defer_close()
        task = asyncio.create_task(stages())

    assert get_current_session_id() is None
    release.set()
    await task

    rows = [row for row in await _rows(ops_engine) if row.pipeline_run_id == run_ids[0]]
    assert len(rows) == 2
    assert all(row.session_id == "chat_1" for row in rows)
