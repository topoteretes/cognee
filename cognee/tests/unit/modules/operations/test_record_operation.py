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
