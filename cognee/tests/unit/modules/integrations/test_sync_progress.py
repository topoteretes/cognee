"""Real SQLite checks for durable progress, ownership, and best-effort reporting."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cognee.modules.integrations import sync_progress as progress
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.pipelines.models.PipelineRun import PipelineRun


@pytest_asyncio.fixture
async def storage(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'progress.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(IntegrationCredential.__table__.create)
        await connection.run_sync(PipelineRun.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        progress, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    yield sessions
    await engine.dispose()


@pytest_asyncio.fixture
async def credential(storage):
    row = IntegrationCredential(
        id=uuid4(),
        user_id=uuid4(),
        workspace_id=uuid4(),
        provider="github",
        provider_account_id="account",
        ciphertext=b"encrypted",
        nonce=b"nonce",
        status="active",
    )
    async with storage() as db:
        db.add(row)
        await db.commit()
    return row


@pytest.mark.asyncio
async def test_live_visibility_and_completed_row(storage, credential):
    async with progress.track_sync("github", credential) as reporter:
        assert progress.get_current_sync_progress() is reporter
        await progress.report_sync_progress(
            phase="downloading", completed=2, total=10, unit="files"
        )
        result = await progress.get_sync_progress(credential)
        assert result.latest.execution_active is True
        assert result.latest.phase_percent == 20
        assert result.latest.phase_eta_seconds is None
        assert result.active_runs[0].run_id == reporter.progress.run_id
    assert progress.get_current_sync_progress() is None
    result = await progress.get_sync_progress(credential)
    assert result.latest.state == "succeeded"
    assert result.latest.execution_active is False
    assert result.latest.ended_at is not None
    assert result.active_runs == []
    async with storage() as db:
        rows = (await db.execute(select(PipelineRun))).scalars().all()
        assert len(rows) == 1
        assert rows[0].operation_name == "integration_sync_progress"
        assert rows[0].pipeline_name is None
        assert rows[0].status is None


@pytest.mark.asyncio
async def test_phase_estimate_resets_unknown_totals_and_waits(storage, credential, monkeypatch):
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    monkeypatch.setattr(progress, "_now", lambda: now)
    async with progress.track_sync("github", credential) as reporter:
        await reporter.report(phase="downloading", completed=0, total=100, unit="files")
        now += timedelta(seconds=10)
        await reporter.report(completed=10)
        assert reporter.snapshot().phase_eta_seconds == 90
        await reporter.report(state="waiting", reason="rate_limited")
        assert reporter.snapshot().phase_eta_seconds is None
        await reporter.report(counts={"scanned": 10})
        assert reporter.snapshot().reason == "rate_limited"
        now += timedelta(seconds=100)
        await reporter.report(state="running", completed=10)
        assert reporter.snapshot().phase_eta_seconds is None
        now += timedelta(seconds=10)
        await reporter.report(completed=20)
        assert reporter.snapshot().phase_eta_seconds == 80
        await reporter.report(phase="indexing")
        view = reporter.snapshot()
        assert view.total is None and view.completed is None and view.unit is None
        assert view.counts == {}
        assert view.phase_percent is None and view.phase_eta_seconds is None
        assert view.phase_elapsed_seconds == 0 and view.elapsed_seconds == 120


@pytest.mark.asyncio
async def test_failure_and_cancellation_do_not_persist_exception_content(storage, credential):
    with pytest.raises(RuntimeError, match="SECRET"):
        async with progress.track_sync("github", credential):
            await progress.report_sync_progress(completed=1, total=1)
            raise RuntimeError("SECRET: private source")
    result = await progress.get_sync_progress(credential)
    assert result.latest.state == "failed"
    assert result.latest.phase_percent is None
    assert "SECRET" not in result.model_dump_json()
    entered = asyncio.Event()

    async def work():
        async with progress.track_sync("github", credential):
            entered.set()
            await asyncio.Future()

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await progress.get_sync_progress(credential)
    assert result.latest.state == "cancelled"
    assert not result.active_runs


@pytest.mark.asyncio
async def test_partial_and_late_updates(storage, credential):
    async with progress.track_sync("github", credential) as reporter:
        await reporter.finish("partial", counts={"failed": 1}, reason="incomplete")
        ended = reporter.snapshot().ended_at
        await reporter.report(phase="indexing", completed=5, total=5)
    result = await progress.get_sync_progress(credential)
    assert result.latest.state == "partial"
    assert result.latest.reason == "incomplete"
    assert result.latest.ended_at == ended
    assert result.latest.phase_percent is None


@pytest.mark.asyncio
async def test_other_process_and_restart_do_not_claim_live_activity(
    storage, credential, monkeypatch
):
    async with progress.track_sync("github", credential) as reporter:
        await reporter.report(completed=2, total=5)
        monkeypatch.setattr(progress, "_active", {})
        result = await progress.get_sync_progress(credential)
        assert result.latest.state == "running"
        assert result.latest.execution_active is None
        assert result.latest.phase_eta_seconds is None
        assert result.active_runs == []
        assert result.unknown_activity_count == 1


@pytest.mark.asyncio
async def test_latest_completion_does_not_hide_older_active_run(storage, credential):
    async with progress.track_sync("github", credential) as first:
        async with progress.track_sync("github", credential) as second:
            result = await progress.get_sync_progress(credential)
            assert len(result.active_runs) == 2
            assert result.latest.run_id == second.progress.run_id
        result = await progress.get_sync_progress(credential)
        assert result.latest.state == "succeeded"
        assert [run.run_id for run in result.active_runs] == [first.progress.run_id]
        assert progress.get_current_sync_progress() is first


@pytest.mark.asyncio
async def test_run_starting_during_status_read_cannot_make_unknown_count_negative(
    storage, credential, monkeypatch
):
    async with progress.track_sync("github", credential):
        second = progress.track_sync("github", credential)
        started = False
        original = AsyncSession.scalar

        async def scalar(db, statement, *args, **kwargs):
            nonlocal started
            result = await original(db, statement, *args, **kwargs)
            if not started and str(statement).startswith("SELECT count("):
                started = True
                await second.__aenter__()
            return result

        monkeypatch.setattr(AsyncSession, "scalar", scalar)
        try:
            result = await progress.get_sync_progress(credential)
            assert started
            assert result.unknown_activity_count == 0
        finally:
            if started:
                await second.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_id", "workspace_id", "status"])
async def test_connection_transfer_or_revocation_blocks_old_reader(storage, credential, field):
    async with progress.track_sync("github", credential):
        pass
    async with storage() as db:
        await db.execute(
            update(IntegrationCredential)
            .where(IntegrationCredential.id == credential.id)
            .values(**{field: "revoked" if field == "status" else uuid4()})
        )
        await db.commit()
    with pytest.raises(CredentialInactiveError):
        await progress.get_sync_progress(credential)
    if field != "status":
        async with storage() as db:
            new_owner = await db.get(IntegrationCredential, credential.id)
        result = await progress.get_sync_progress(new_owner)
        assert result.latest is None


@pytest.mark.asyncio
async def test_distinct_accounts_and_unrelated_ledger_rows_are_private(storage, credential):
    async with progress.track_sync("github", credential):
        pass
    async with storage() as db:
        other = IntegrationCredential(
            id=uuid4(),
            user_id=credential.user_id,
            workspace_id=credential.workspace_id,
            provider="github",
            provider_account_id="another",
            ciphertext=b"a",
            nonce=b"b",
        )
        db.add(other)
        db.add(
            PipelineRun(
                user_id=credential.user_id, operation_name="recall", run_info={"secret": "text"}
            )
        )
        await db.commit()
    result = await progress.get_sync_progress(other)
    assert result.latest is None
    assert result.active_runs == []


@pytest.mark.asyncio
async def test_io_failure_does_not_change_result_and_sink_is_independent(
    credential, monkeypatch, caplog
):
    def unavailable():
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(progress, "get_relational_engine", unavailable)
    sink = AsyncMock()
    async with progress.track_sync("github", credential, sink=sink) as reporter:
        await reporter.report(completed=3)
    assert sink.await_count == 2
    assert sink.call_args.args[0].state == "succeeded"
    sink.side_effect = RuntimeError("sink unavailable")
    with pytest.raises(ValueError, match="original"):
        async with progress.track_sync("github", credential, sink=sink):
            raise ValueError("original")
    assert "sink unavailable" not in caplog.text
    assert "storage unavailable" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [
        {"completed": -1},
        {"completed": True},
        {"total": "4"},
        {"completed": 3, "total": 2},
        {"counts": {"failed": -1}},
        {"counts": {"failed_ingestion": True}},
        {"reason": "secret exception"},
        {"phase": "private file name"},
    ],
)
async def test_invalid_programmer_input_fails_before_writing(storage, credential, update):
    async with progress.track_sync("github", credential) as reporter:
        before = reporter.snapshot().updated_at
        with pytest.raises((ValueError, ValidationError)):
            await reporter.report(**update)
        assert reporter.snapshot().updated_at == before


@pytest.mark.asyncio
async def test_duplicates_do_not_write_and_sink_cannot_mutate_owned_state(storage, credential):
    sink = AsyncMock()
    async with progress.track_sync("github", credential, sink=sink) as reporter:
        await reporter.report(counts={"scanned": 3})
        await reporter.flush()
        assert sink.await_count == 2
        sink.call_args.args[0].counts["scanned"] = 100
        await reporter.report(counts={"scanned": 3})
        await reporter.flush()
        assert sink.await_count == 2
        assert reporter.snapshot().counts == {"scanned": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("block_at", ["storage", "sink"])
async def test_slow_delivery_coalesces_without_blocking_progress(
    storage, credential, monkeypatch, block_at
):
    entered, release = asyncio.Event(), asyncio.Event()
    observations = []

    async def pause(snapshot):
        if snapshot.completed == 1:
            entered.set()
            await release.wait()

    async def sink(snapshot):
        if block_at == "sink":
            await pause(snapshot)
        observations.append(snapshot)

    original = progress.SyncProgressReporter._publish

    async def publish(reporter, snapshot=None):
        if block_at == "storage" and snapshot is not None:
            await pause(snapshot)
        await original(reporter, snapshot)

    monkeypatch.setattr(progress.SyncProgressReporter, "_publish", publish)
    async with progress.track_sync("github", credential, sink=sink) as reporter:
        await reporter.report(completed=1, total=100)
        await asyncio.wait_for(entered.wait(), timeout=5)
        try:
            for completed in range(2, 101):
                await reporter.report(completed=completed)
            result = await progress.get_sync_progress(credential)
            assert result.latest.completed == 100
            assert result.latest.execution_active is True
            finish = asyncio.create_task(reporter.finish())
            await asyncio.sleep(0)
            assert not finish.done()
        finally:
            release.set()
        await finish
        assert reporter._writer.done()
        assert reporter._pending is None
    assert [item.completed for item in observations] == [None, 1, 100]
    assert observations[-1].state == "succeeded"
    result = await progress.get_sync_progress(credential)
    assert result.latest.state == "succeeded"
    assert result.latest.completed == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_flush", [False, True])
async def test_repeated_cancellation_joins_writer_and_preserves_final_observation(
    storage, credential, cancel_during_flush
):
    entered, release, body_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    reporters = []

    async def sink(snapshot):
        if snapshot.completed == 1 and snapshot.state == "running":
            entered.set()
            await release.wait()

    async def work():
        async with progress.track_sync("github", credential, sink=sink) as reporter:
            reporters.append(reporter)
            await reporter.report(completed=1)
            await entered.wait()
            body_done.set()
            if not cancel_during_flush:
                await asyncio.Future()

    task = asyncio.create_task(work())
    await asyncio.wait_for(body_done.wait(), timeout=5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    reporter = reporters[0]
    assert reporter._writer.done()
    assert reporter._pending is None
    assert reporter.progress.run_id not in progress._active
    result = await progress.get_sync_progress(credential)
    assert result.latest.state == ("succeeded" if cancel_during_flush else "cancelled")
    assert result.latest.execution_active is False


@pytest.mark.asyncio
async def test_no_context_is_noop():
    await progress.report_sync_progress(completed=3)
    assert progress.get_current_sync_progress() is None


@pytest.mark.asyncio
async def test_multiple_unknown_and_active_runs_are_counted_without_history_scan(
    storage, credential, monkeypatch
):
    async with (
        progress.track_sync("github", credential) as first,
        progress.track_sync("github", credential) as second,
        progress.track_sync("github", credential),
        progress.track_sync("github", credential),
    ):
        monkeypatch.setattr(
            progress,
            "_active",
            {
                first.progress.run_id: first,
                second.progress.run_id: second,
            },
        )
        result = await progress.get_sync_progress(credential)
        assert result.unknown_activity_count == 2
        assert len(result.active_runs) == 2
        assert result.latest.execution_active is None


@pytest.mark.asyncio
async def test_provider_diagnostic_counters_are_retained(storage, credential):
    async with progress.track_sync("github", credential) as reporter:
        await reporter.finish("partial", counts={"failed_ingestion": 2, "requests": 4})
    result = await progress.get_sync_progress(credential)
    assert result.latest.counts == {"failed_ingestion": 2, "requests": 4}


@pytest.mark.asyncio
async def test_updates_within_a_phase_are_throttled_and_changes_publish_at_once(
    storage, credential, monkeypatch
):
    monkeypatch.setattr(progress, "PUBLISH_INTERVAL_SECONDS", 0.2)
    sink = AsyncMock()
    async with progress.track_sync("github", credential, sink=sink) as reporter:
        await reporter.flush()
        sink.reset_mock()
        await reporter.report(phase="downloading", completed=0, total=100, unit="files")
        await reporter.flush()
        sink.reset_mock()
        for done in range(1, 50):
            await reporter.report(completed=done)
            await asyncio.sleep(0)
        await reporter.flush()
        assert sink.await_count <= 2
        assert sink.call_args.args[0].completed == 49
        sink.reset_mock()
        await reporter.report(phase="processing")
        await reporter.flush()
        assert sink.call_args.args[0].phase == "processing"
