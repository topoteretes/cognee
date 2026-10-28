"""Policy gates and wall-clock schedules, without GitHub or ingestion services."""

import asyncio
import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.modules.integrations.github.sync_settings import (
    NEXT_SYNC_KEY,
    SETTINGS_KEY,
    GithubRepositorySettings,
    GithubSyncSettings,
    SyncSchedule,
    next_sync_at,
    within_schedule,
)

sync = importlib.import_module("cognee.modules.integrations.github.sync")
credentials = importlib.import_module("cognee.modules.integrations.credentials")


def dt(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def scheduled(**overrides):
    return GithubSyncSettings(
        mode="scheduled",
        schedule=SyncSchedule(
            interval_minutes=15,
            timezone="Europe/Berlin",
            weekdays=[0, 1, 2, 3, 4],
            start_time="09:00",
            end_time="17:00",
            **overrides,
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"interval_minutes": True},
        {"interval_minutes": 0},
        {"interval_minutes": "15"},
        {"interval_minutes": 15, "weekdays": []},
        {"interval_minutes": 15, "weekdays": [7]},
        {"interval_minutes": 15, "timezone": "Missing/Zone"},
        {"interval_minutes": 15, "start_time": "17:00", "end_time": "09:00"},
        {"interval_minutes": 15, "start_time": "09:00"},
    ],
)
def test_invalid_schedules_fail(payload):
    with pytest.raises(ValidationError):
        SyncSchedule(**payload)


def test_documents_fail_instead_of_promising_unsafe_removal():
    with pytest.raises(ValidationError, match="source-owned"):
        GithubRepositorySettings(include_documents=True)


def test_friday_end_moves_to_monday_after_dst_change():
    assert next_sync_at(scheduled(), dt("2026-10-23T15:00")) == dt("2026-10-26T08:00")
    assert not within_schedule(scheduled(), dt("2026-10-24T10:00"))
    assert not within_schedule(scheduled(), dt("2026-10-23T15:00"))


def test_custom_interval_and_exclusive_window():
    settings = GithubSyncSettings(
        mode="scheduled",
        schedule=SyncSchedule(
            interval_minutes=37,
            start_time="09:00",
            end_time="10:00",
        ),
    )
    assert next_sync_at(settings, dt("2026-10-26T09:00")) == dt("2026-10-26T09:37")
    assert next_sync_at(settings, dt("2026-10-26T09:37")) == dt("2026-10-27T09:00")


def test_dst_gap_skipped_and_fold_keeps_real_slots():
    settings = GithubSyncSettings(
        mode="scheduled",
        schedule=SyncSchedule(
            interval_minutes=30,
            timezone="Europe/Berlin",
            start_time="02:00",
            end_time="03:00",
        ),
    )
    assert next_sync_at(settings, dt("2026-03-29T00:59")) == dt("2026-03-30T00:00")
    assert next_sync_at(settings, dt("2026-10-25T00:30")) == dt("2026-10-25T01:00")


@pytest.fixture
def orchestration(monkeypatch):
    progress = importlib.import_module("cognee.modules.integrations.sync_progress")
    monkeypatch.setattr(progress.SyncProgressReporter, "_publish", AsyncMock())
    monkeypatch.setattr(credentials, "record_sync_result", AsyncMock())
    row = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        workspace_id=None,
        provider="github",
        provider_account_id="42",
        status="active",
        provider_metadata={},
    )
    monkeypatch.setattr(
        credentials, "require_active_credential", AsyncMock(side_effect=lambda _: row)
    )

    async def patch_metadata(provider, account, patch):
        row.provider_metadata.update(patch)
        return row

    monkeypatch.setattr(
        credentials, "update_provider_metadata", AsyncMock(side_effect=patch_metadata)
    )
    monkeypatch.setattr(sync, "mint_installation_token", AsyncMock(return_value=("token", None)))
    monkeypatch.setattr(
        sync,
        "list_installation_repository_records",
        AsyncMock(
            return_value=[
                {"id": 1, "full_name": "org/one", "default_branch": "main"},
                {"id": 2, "full_name": "org/two", "default_branch": "main"},
            ]
        ),
    )
    settings_module = importlib.import_module("cognee.modules.integrations.github.sync_settings")
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    db.execute.return_value = result
    engine = MagicMock()
    engine.get_async_session.return_value.__aenter__.return_value = db
    monkeypatch.setattr(settings_module, "get_relational_engine", lambda: engine)
    build = AsyncMock(return_value="completed")
    monkeypatch.setattr(sync, "_sync_repositories", build)
    return row, build


@pytest.mark.asyncio
async def test_paused_manual_sync_and_push_gates(orchestration):
    row, build = orchestration
    row.provider_metadata[SETTINGS_KEY] = {"enabled": False}
    assert await sync.sync_repositories(row, trigger="push") == "paused"
    build.assert_not_awaited()
    assert await sync.sync_repositories(row, trigger="manual") == "completed"
    build.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_branch_and_selection_are_applied(orchestration):
    row, build = orchestration
    row.provider_metadata[SETTINGS_KEY] = {
        "repositories": {"1": {"branch": "release", "include_paths": ["src/**"]}}
    }
    await sync.sync_repositories(row, ["org/one"], trigger="push", pushed_branch="release")
    assert build.call_args.args[1] == ["org/one"]
    assert build.call_args.kwargs["policies"]["org/one"].include_paths == ["src/**"]
    await sync.sync_repositories(row, ["org/one"], trigger="push", pushed_branch="main")
    assert build.call_args.args[1] == []


@pytest.mark.asyncio
async def test_overdue_weekend_does_not_run_and_tick_claims_once(orchestration):
    row, build = orchestration
    row.provider_metadata.update(
        {SETTINGS_KEY: scheduled().model_dump(), NEXT_SYNC_KEY: "2026-10-23T14:45:00+00:00"}
    )
    assert (
        await sync.sync_repositories(row, trigger="scheduled", now=dt("2026-10-24T10:00"))
        == "outside_schedule"
    )
    build.assert_not_awaited()
    assert (
        await sync.sync_repositories(row, trigger="scheduled", now=dt("2026-10-26T08:01"))
        == "completed"
    )
    assert (
        await sync.sync_repositories(row, trigger="scheduled", now=dt("2026-10-26T08:01"))
        == "not_due"
    )
    build.assert_awaited_once()


@pytest.mark.asyncio
async def test_queued_push_runs_after_current_sync(orchestration):
    row, build = orchestration
    started, release = asyncio.Event(), asyncio.Event()

    async def hold(*args, **kwargs):
        started.set()
        await release.wait()
        return "completed"

    build.side_effect = hold
    first = asyncio.create_task(sync.sync_repositories(row, trigger="push"))
    await started.wait()
    second = asyncio.create_task(sync.sync_repositories(row, trigger="push"))
    await asyncio.sleep(0)
    assert build.await_count == 1
    release.set()
    await asyncio.gather(first, second)
    assert build.await_count == 2


@pytest.mark.asyncio
async def test_empty_selection_still_requests_retirement(orchestration):
    row, build = orchestration
    row.provider_metadata[SETTINGS_KEY] = {"repositories": {}}
    await sync.sync_repositories(row)
    assert build.call_args.args[1] == []
    assert build.call_args.kwargs["retire_selection"] == set()


@pytest.mark.asyncio
async def test_pause_while_waiting_prevents_queued_push(orchestration):
    row, build = orchestration
    started, release = asyncio.Event(), asyncio.Event()

    async def hold(*args, **kwargs):
        started.set()
        await release.wait()
        return "completed"

    build.side_effect = hold
    first = asyncio.create_task(sync.sync_repositories(row, trigger="push"))
    await started.wait()
    second = asyncio.create_task(sync.sync_repositories(row, trigger="push"))
    await asyncio.sleep(0)
    row.provider_metadata[SETTINGS_KEY] = {"enabled": False}
    release.set()
    assert await first == "completed"
    assert await second == "paused"
    assert build.await_count == 1


@pytest.mark.asyncio
async def test_initial_waits_for_selected_schedule(orchestration):
    row, build = orchestration
    settings = scheduled()
    settings.first_sync = "scheduled"
    row.provider_metadata[SETTINGS_KEY] = settings.model_dump()
    assert await sync.sync_repositories(row, trigger="initial") == "mode_skipped"
    build.assert_not_awaited()


def test_naive_clock_and_invalid_paths_fail():
    with pytest.raises(ValueError, match="timezone-aware"):
        naive = dt("2026-10-26T00:00").replace(tzinfo=None)
        next_sync_at(scheduled(), naive)
    with pytest.raises(ValidationError):
        GithubRepositorySettings(include_paths=["../outside"])


@pytest.mark.asyncio
async def test_settings_save_requires_ack_and_preserves_metadata(monkeypatch):
    from unittest.mock import MagicMock

    from cognee.modules.integrations.github import sync_settings as module

    row = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        workspace_id=None,
        provider="github",
        provider_account_id="42",
        provider_metadata={"account_login": "org"},
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    db.execute.return_value = result
    engine = MagicMock()
    engine.get_async_session.return_value.__aenter__.return_value = db
    monkeypatch.setattr(module, "get_relational_engine", lambda: engine)
    retire = importlib.import_module("cognee.modules.integrations.github.retire")
    monkeypatch.setattr(retire, "assert_tracked_selection", AsyncMock())
    settings = GithubSyncSettings(repositories={"1": {"include_paths": ["src/**"]}})
    with pytest.raises(ValueError, match="acknowledge_removal"):
        await module.save_sync_settings(row, settings)
    db.commit.assert_not_awaited()
    await module.save_sync_settings(row, settings, acknowledge_removal=True)
    assert row.provider_metadata["account_login"] == "org"
    assert row.provider_metadata[SETTINGS_KEY]["repositories"]["1"]["include_paths"] == ["src/**"]
    assert row.provider_metadata[NEXT_SYNC_KEY] is None
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_waiting_scheduled_run_checks_clock_after_lock(orchestration, monkeypatch):
    row, build = orchestration
    row.provider_metadata.update(
        {
            SETTINGS_KEY: scheduled().model_dump(),
            NEXT_SYNC_KEY: "2026-10-26T15:45:00+00:00",
        }
    )
    started, release = asyncio.Event(), asyncio.Event()
    clock = [dt("2026-10-26T15:59")]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr(sync, "datetime", Clock)

    async def hold(*args, **kwargs):
        started.set()
        await release.wait()
        return "completed"

    build.side_effect = hold
    first = asyncio.create_task(sync.sync_repositories(row))
    await started.wait()
    queued = asyncio.create_task(sync.sync_repositories(row, trigger="scheduled"))
    await asyncio.sleep(0)
    clock[0] = dt("2026-10-26T16:01")
    release.set()
    assert await first == "completed"
    assert await queued == "outside_schedule"
    assert build.await_count == 1


@pytest.mark.asyncio
async def test_a_selected_repository_that_lost_access_is_skipped_not_fatal(orchestration, caplog):
    row, build = orchestration
    row.provider_metadata[SETTINGS_KEY] = {"repositories": {"1": {}, "3": {}}}
    with caplog.at_level("WARNING"):
        assert await sync.sync_repositories(row) == "completed"
    assert list(build.await_args.args[1]) == ["org/one"]
    # The saved selection keeps the missing id, and the ledger therefore keeps its content.
    assert build.await_args.kwargs["retire_selection"] == {"1", "3"}
    assert "3" in caplog.text
    credentials.record_sync_result.assert_awaited_once_with(row, status="ok")


@pytest.mark.asyncio
async def test_an_empty_selection_leaves_the_last_attempt_unrecorded(orchestration):
    row, build = orchestration
    build.return_value = "nothing_selected"
    assert await sync.sync_repositories(row) == "nothing_selected"
    credentials.record_sync_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_result_status_records_partial_attempt(orchestration):
    row, build = orchestration
    build.return_value = "partial"
    assert await sync.sync_repositories(row) == "partial"
    credentials.record_sync_result.assert_awaited_once_with(row, status="degraded")


@pytest.mark.asyncio
async def test_missing_default_branch_fails_explicitly(orchestration):
    row, build = orchestration
    sync.list_installation_repository_records.return_value = [{"id": 1, "full_name": "org/one"}]
    with pytest.raises(ValueError, match="no default branch"):
        await sync.sync_repositories(row)
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_due_runner_continues_after_account_failure_without_freezing_clock(monkeypatch):
    from cognee.modules.integrations.github import sync_settings as module

    rows = [SimpleNamespace(provider_account_id="1"), SimpleNamespace(provider_account_id="2")]
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    db.execute.return_value = result
    engine = MagicMock()
    engine.get_async_session.return_value.__aenter__.return_value = db
    monkeypatch.setattr(module, "get_relational_engine", lambda: engine)
    run = AsyncMock(side_effect=[RuntimeError("private detail"), "completed"])
    monkeypatch.setattr(sync, "sync_repositories", run)
    assert await module.run_due_github_syncs() == {"1": "failed", "2": "completed"}
    assert run.await_count == 2
    assert all(call.kwargs["now"] is None for call in run.await_args_list)
