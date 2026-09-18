"""Contract tests for ``record_operation`` (SDK-399).

A non-pipeline operation wrapped in ``record_operation(name)`` must leave
exactly one durable ``pipeline_runs`` row with ``status = NULL`` carrying
operation name, triggering user/tenant, start+end timestamps on the single
record, and an unambiguous outcome ("succeeded"/"failed" + error class).
The recorder must never break the operation it records, and NULL-status
rows must stay invisible to the legacy latest-row status readers.

Runs against a real temporary SQLite database — no LLM, no network.
"""

import asyncio
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.pipelines.models.PipelineRun import PipelineRun, PipelineRunStatus

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")
get_pipeline_status_mod = importlib.import_module(
    "cognee.modules.pipelines.operations.get_pipeline_status"
)
# get_pipeline_status delegates its query to this module now, so the real
# DB call this test needs to intercept happens here, not in either module
# above.
get_pipeline_run_by_dataset_mod = importlib.import_module(
    "cognee.modules.pipelines.methods.get_pipeline_run_by_dataset"
)

record_operation = record_operation_mod.record_operation
get_current_operation = record_operation_mod.get_current_operation


@pytest_asyncio.fixture
async def ops_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="ops_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[PipelineRun.__table__])

    for module in (record_operation_mod, get_pipeline_run_by_dataset_mod):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _fetch_rows(engine):
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRun).order_by(PipelineRun.created_at))
        return result.scalars().all()


def _fake_user():
    return SimpleNamespace(id=uuid4(), tenant_id=uuid4())


@pytest.mark.asyncio
async def test_successful_operation_writes_one_self_contained_row(ops_engine):
    """The ticket's non-pipeline end-to-end case at unit grain."""
    user = _fake_user()
    dataset_id = uuid4()

    async with record_operation("search", user=user, dataset_id=dataset_id):
        pass

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    row = rows[0]

    assert row.operation_name == "search"
    assert row.user_id == user.id
    assert row.tenant_id == user.tenant_id
    assert row.dataset_id == dataset_id
    assert row.outcome == "succeeded"
    assert row.error_class is None
    # Start and end are readable from this single record, no self-join.
    assert row.started_at is not None
    assert row.ended_at is not None
    assert row.started_at <= row.ended_at
    # Non-pipeline rows are status-NULL and never touch run_info.
    assert row.status is None
    assert row.pipeline_name is None
    assert row.pipeline_id is None
    assert row.run_info is None
    assert row.pipeline_run_id is not None
    # Tokens measured (zero) rather than unmeasured (NULL).
    assert row.tokens_in == 0
    assert row.tokens_out == 0


@pytest.mark.asyncio
async def test_failed_operation_records_outcome_and_error_class(ops_engine):
    """The ticket's failed-op case: exception propagates AND a failed row lands."""
    user = _fake_user()

    with pytest.raises(ValueError, match="boom"):
        async with record_operation("forget", user=user):
            raise ValueError("boom")

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    row = rows[0]

    assert row.operation_name == "forget"
    assert row.outcome == "failed"
    assert row.error_class == "ValueError"
    assert row.ended_at is not None
    assert row.user_id == user.id


@pytest.mark.asyncio
async def test_operation_without_user_or_dataset_records_nulls(ops_engine):
    """prune-style operations: record with NULL user, never invent one."""
    async with record_operation("prune_data"):
        pass

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    assert rows[0].user_id is None
    assert rows[0].tenant_id is None
    assert rows[0].dataset_id is None
    assert rows[0].outcome == "succeeded"


@pytest.mark.asyncio
async def test_persistence_failure_never_breaks_the_operation(ops_engine, monkeypatch):
    """Recorder errors are logged and swallowed; the body's result survives."""

    def _broken_engine():
        raise RuntimeError("relational database is gone")

    monkeypatch.setattr(record_operation_mod, "get_relational_engine", _broken_engine)

    completed = False
    async with record_operation("prune_system"):
        completed = True

    assert completed


@pytest.mark.asyncio
async def test_missing_store_logs_debug_not_warning(ops_engine, monkeypatch, caplog):
    """An unreachable relational store must not splash a warning traceback.

    Prune deletes the very database the ledger writes to, and nothing exists
    before setup() — both are normal in the quickstart examples, so the
    skipped write logs at debug only."""
    from sqlalchemy.exc import OperationalError

    def _store_gone():
        raise OperationalError("stmt", None, Exception("unable to open database file"))

    monkeypatch.setattr(record_operation_mod, "get_relational_engine", _store_gone)

    with caplog.at_level("DEBUG"):
        async with record_operation("prune_data"):
            pass

    warnings = [
        r for r in caplog.records if r.levelname == "WARNING" and "persist" in r.getMessage()
    ]
    assert warnings == []
    debugs = [
        r
        for r in caplog.records
        if r.levelname == "DEBUG" and "relational store unavailable" in r.getMessage()
    ]
    assert len(debugs) == 1


@pytest.mark.asyncio
async def test_unexpected_persist_failure_still_warns(ops_engine, monkeypatch, caplog):
    """Only the store-unavailable class is quiet; other failures stay loud."""

    def _broken_engine():
        raise RuntimeError("relational database is gone")

    monkeypatch.setattr(record_operation_mod, "get_relational_engine", _broken_engine)

    with caplog.at_level("DEBUG"):
        async with record_operation("prune_data"):
            pass

    warnings = [
        r for r in caplog.records if r.levelname == "WARNING" and "persist" in r.getMessage()
    ]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_persistence_failure_does_not_mask_operation_error(ops_engine, monkeypatch):
    """When both the operation and the write fail, the operation's error wins."""

    def _broken_engine():
        raise RuntimeError("relational database is gone")

    monkeypatch.setattr(record_operation_mod, "get_relational_engine", _broken_engine)

    with pytest.raises(KeyError):
        async with record_operation("forget"):
            raise KeyError("missing")


@pytest.mark.asyncio
async def test_late_binding_via_context_and_nested_coroutine(ops_engine):
    """set_user inside the scope and get_current_operation from a child both persist."""
    user = _fake_user()
    dataset_id = uuid4()

    async def _deep_call_site():
        # Deep call sites (e.g. recall's lazy user resolution) bind through
        # the contextvar, without signature plumbing.
        operation = get_current_operation()
        assert operation is not None
        operation.set_dataset(dataset_id)

    async with record_operation("recall") as ctx:
        ctx.set_user(user)
        await asyncio.create_task(_deep_call_site())

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    assert rows[0].user_id == user.id
    assert rows[0].tenant_id == user.tenant_id
    assert rows[0].dataset_id == dataset_id


@pytest.mark.asyncio
async def test_current_operation_context_is_reset_on_exit(ops_engine):
    assert get_current_operation() is None
    async with record_operation("search"):
        assert get_current_operation() is not None
    assert get_current_operation() is None


@pytest.mark.asyncio
async def test_operation_rows_are_invisible_to_pipeline_status_readers(ops_engine):
    """Consumer non-regression: NULL-status rows never shadow pipeline state."""
    dataset_id = uuid4()

    async with ops_engine.get_async_session() as session:
        session.add(
            PipelineRun(
                pipeline_run_id=uuid4(),
                pipeline_name="cognify_pipeline",
                pipeline_id=uuid4(),
                status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                dataset_id=dataset_id,
                run_info={},
            )
        )
        await session.commit()

    # A newer operation row on the SAME dataset must not become the rn=1 row.
    async with record_operation("search", user=_fake_user(), dataset_id=dataset_id):
        pass

    statuses = await get_pipeline_status_mod.get_pipeline_status([dataset_id], "cognify_pipeline")

    assert statuses == {str(dataset_id): PipelineRunStatus.DATASET_PROCESSING_COMPLETED}


@pytest.mark.asyncio
async def test_deferred_close_writes_nothing_until_finish_operation(ops_engine):
    """A deferred operation's row lands when the background work ends, not at launch."""
    async with record_operation("improve", user=_fake_user()) as operation:
        operation.defer_close()

    assert await _fetch_rows(ops_engine) == []  # the scope exit wrote nothing

    await record_operation_mod.finish_operation(operation)

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    assert rows[0].operation_name == "improve"
    assert rows[0].outcome == "succeeded"
    assert rows[0].started_at <= rows[0].ended_at


@pytest.mark.asyncio
async def test_finish_operation_records_the_error(ops_engine):
    async with record_operation("improve", user=_fake_user()) as operation:
        operation.defer_close()

    await record_operation_mod.finish_operation(operation, error=RuntimeError("boom"))

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    assert rows[0].outcome == "failed"
    assert rows[0].error_class == "RuntimeError"


@pytest.mark.asyncio
async def test_cancelled_deferred_work_still_records_a_failed_row(ops_engine):
    """The improve ``_run_detached`` pattern: a cancelled background run must
    not vanish — with the close deferred, not even a launch row exists, so the
    write has to happen from the cancelled task's own cleanup."""
    started = asyncio.Event()

    async def detached(operation):
        error: BaseException | None = None
        try:
            started.set()
            await asyncio.Event().wait()  # blocks until cancelled
        except BaseException as caught:
            error = caught
            raise
        finally:
            await record_operation_mod.finish_operation(operation, error=error)

    async with record_operation("improve", user=_fake_user()) as operation:
        operation.defer_close()
    task = asyncio.create_task(detached(operation))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    assert rows[0].outcome == "failed"
    assert rows[0].error_class == "CancelledError"


@pytest.mark.asyncio
async def test_outcome_override_marks_a_clean_exit_failed(ops_engine):
    """set_outcome lets an operation record failure its body did not raise."""
    from cognee.modules.pipelines.models import OperationOutcome

    async with record_operation("improve", user=_fake_user()) as operation:
        operation.set_outcome(OperationOutcome.FAILED)

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    assert rows[0].outcome == "failed"


@pytest.mark.asyncio
async def test_raised_exception_wins_over_the_override(ops_engine):
    from cognee.modules.pipelines.models import OperationOutcome

    with pytest.raises(ValueError, match="boom"):
        async with record_operation("improve", user=_fake_user()) as operation:
            operation.set_outcome(OperationOutcome.SUCCEEDED)
            raise ValueError("boom")

    rows = await _fetch_rows(ops_engine)
    assert rows[0].outcome == "failed"
    assert rows[0].error_class == "ValueError"
