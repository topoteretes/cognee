"""Sync progress observes ingestion without changing its source and cursor contract."""

import asyncio
import importlib
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import CredentialInactiveError

CREDENTIAL = SimpleNamespace(user_id="owner", provider_account_id="account")


@pytest.fixture
def progress(monkeypatch):
    from cognee.modules.integrations import sync_progress

    events = []
    finishes = []

    async def report(**kwargs):
        events.append(deepcopy(kwargs))

    async def finish(state, **kwargs):
        finishes.append((state, deepcopy(kwargs)))

    reporter = SimpleNamespace(report=report, finish=finish)

    @asynccontextmanager
    async def track(*args, **kwargs):
        try:
            yield reporter
        except asyncio.CancelledError:
            await finish("cancelled")
            raise

    monkeypatch.setattr(sync_progress, "track_sync", track)
    monkeypatch.setattr(sync_progress, "report_sync_progress", report)
    monkeypatch.setattr(sync_progress, "get_current_sync_progress", lambda: reporter)
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda value: value)
    )
    record = AsyncMock()
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", record)
    monkeypatch.setattr("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
    return SimpleNamespace(events=events, finishes=finishes, record=record, reporter=reporter)


@pytest.mark.asyncio
async def test_live_scan_counts_are_visible_before_scope_finishes_without_double_counting(
    monkeypatch, progress
):
    scanned = asyncio.Event()
    release = asyncio.Event()
    sources = []

    def make_source(scope, name, checkpoint):
        source = SimpleNamespace(cognee_sync_stats={}, checkpoint=checkpoint)
        sources.append(source)
        return source

    async def remember(source, **kwargs):
        source.cognee_sync_stats["scanned"] = 2
        # run_in_executor does not copy contextvars, unlike asyncio.to_thread.
        await asyncio.get_running_loop().run_in_executor(None, source.checkpoint)
        if len(sources) == 1:
            scanned.set()
            await release.wait()
        source.cognee_sync_stats["scanned"] = 3
        await asyncio.get_running_loop().run_in_executor(None, source.checkpoint)
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.remember"), "remember", remember
    )
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    task = asyncio.create_task(
        ingestion.sync_scopes(
            "google_drive",
            CREDENTIAL,
            counts,
            scopes=["one", "two"],
            dataset_name="drive",
            make_source=make_source,
        )
    )
    await asyncio.wait_for(scanned.wait(), timeout=5)
    try:
        assert not task.done()
        assert progress.events[-1]["counts"]["scanned"] == 2
        assert counts["scanned"] == 0
        assert progress.events[0] == {
            "phase": "processing",
            "completed": 0,
            "total": 2,
            "unit": "scopes",
            "counts": {"scanned": 0, "skipped": 0, "failed": 0},
        }
    finally:
        release.set()
        await task
    assert counts["scanned"] == 6
    assert progress.events[-1]["completed"] == 2
    assert progress.events[-1]["counts"]["scanned"] == 6
    assert any(event.get("counts", {}).get("scanned") == 5 for event in progress.events)
    assert all("processed" not in event.get("counts", {}) for event in progress.events)


@pytest.mark.asyncio
async def test_single_source_reports_scans_without_inventing_a_total(monkeypatch, progress):
    source = SimpleNamespace(cognee_sync_stats={})

    def make_source(**kwargs):
        source.checkpoint = kwargs["check_active"]
        return source

    async def remember(source, **kwargs):
        source.cognee_sync_stats["scanned"] = 7
        await asyncio.get_running_loop().run_in_executor(None, source.checkpoint)
        assert progress.events[-1]["counts"]["scanned"] == 7
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(ingestion, "source_factory", lambda _: make_source)
    monkeypatch.setattr(ingestion, "build_service", lambda *args: object())
    monkeypatch.setattr(ingestion, "retire_resources", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.remember"), "remember", remember
    )
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    status, counts = await ingestion.sync_selected_resources(
        "gmail",
        CREDENTIAL,
        counts,
        selection=["INBOX"],
        selection_kwarg="label_ids",
        dataset_name="gmail",
        access_token_for=AsyncMock(return_value="token"),
    )
    assert status == "ok" and counts["scanned"] == 7
    assert all("total" not in event for event in progress.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,failed,expected",
    [("ok", 0, "succeeded"), ("degraded", 0, "partial"), ("ok", 2, "partial")],
)
async def test_terminal_state_includes_handled_failures(progress, status, failed, expected):
    async def sync_source(credential, counts):
        counts.update(scanned=4, failed=failed)
        return status, counts

    await ingestion.run_sync("gmail", CREDENTIAL, sync_source)
    assert progress.events[0]["phase"] == "checking"
    assert progress.finishes[-1][0] == expected
    assert progress.finishes[-1][1]["counts"]["failed"] == failed
    assert progress.record.await_args.kwargs["status"] == status
    assert not ingestion.sync_is_running("gmail", "account")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,reason",
    [
        (ingestion.RateLimitedError("secret token"), "rate_limited"),
        (CredentialInactiveError(), "connection_changed"),
        (RuntimeError("secret document"), "provider_error"),
    ],
)
async def test_failure_reason_is_safe_and_original_error_is_preserved(progress, error, reason):
    async def sync_source(credential, counts):
        raise error

    with pytest.raises(type(error)) as caught:
        await ingestion.run_sync("gmail", CREDENTIAL, sync_source)
    assert caught.value is error
    assert progress.finishes[-1][0] == "failed"
    assert progress.finishes[-1][1]["reason"] == reason
    assert "secret" not in repr(progress.finishes)
    assert not ingestion.sync_is_running("gmail", "account")


@pytest.mark.asyncio
async def test_duplicate_keeps_existing_run_and_cancellation_releases_guard(progress):
    entered = asyncio.Event()

    async def sync_source(credential, counts):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(ingestion.run_sync("gmail", CREDENTIAL, sync_source))
    await asyncio.wait_for(entered.wait(), timeout=5)
    second = AsyncMock()
    try:
        await ingestion.run_sync("gmail", CREDENTIAL, second)
        second.assert_not_awaited()
        assert len(progress.events) == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert progress.finishes[-1][0] == "cancelled"
    assert not ingestion.sync_is_running("gmail", "account")
    progress.record.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkpoint_keeps_disconnect_authoritative(progress):
    source = SimpleNamespace(cognee_sync_stats={"scanned": 3})
    bridge = ingestion._SourceProgress(CREDENTIAL, {})
    bridge.source = source
    checkpoint = bridge.checkpoint
    ingestion.require_active_credential.side_effect = CredentialInactiveError()
    with pytest.raises(CredentialInactiveError):
        await asyncio.get_running_loop().run_in_executor(None, checkpoint)
    assert progress.events == []


@pytest.mark.asyncio
async def test_checkpoint_survives_reporting_failure(progress):
    async def broken(**kwargs):
        raise RuntimeError("telemetry down")

    progress.reporter.report = broken
    bridge = ingestion._SourceProgress(CREDENTIAL, {})
    bridge.source = SimpleNamespace(cognee_sync_stats={"scanned": 3})
    await asyncio.get_running_loop().run_in_executor(None, bridge.checkpoint)


@pytest.mark.asyncio
async def test_unavailable_progress_storage_and_sink_do_not_change_success(monkeypatch):
    from uuid import uuid4

    from cognee.modules.integrations import sync_progress

    credential = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        workspace_id=None,
        provider="gmail",
        provider_account_id="account",
    )
    snapshots = []
    track = sync_progress.track_sync

    async def unavailable_sink(snapshot):
        snapshots.append(snapshot)
        raise OSError("status consumer unavailable")

    def unavailable_database():
        raise OSError("status storage unavailable")

    def tracked(provider, credential):
        return track(provider, credential, sink=unavailable_sink)

    monkeypatch.setattr(sync_progress, "track_sync", tracked)
    monkeypatch.setattr(sync_progress, "get_relational_engine", unavailable_database)
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda value: value)
    )
    record = AsyncMock()
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", record)

    async def sync_source(credential, counts):
        counts.update(scanned=9, skipped=2, skipped_unsupported=2)
        return "ok", counts

    await ingestion.run_sync("gmail", credential, sync_source)
    assert snapshots[-1].state == "succeeded"
    assert snapshots[-1].counts["skipped_unsupported"] == 2
    assert record.await_args.kwargs["status"] == "ok"
    assert not ingestion.sync_is_running("gmail", "account")
