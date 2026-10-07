"""Idle-session consolidation: bridge session memory before the cache TTL expires it."""

import asyncio
import importlib
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.modules.improve.config import ImproveConfig

consolidation = importlib.import_module("cognee.modules.improve.idle_consolidation")
improve_package = importlib.import_module("cognee.api.v1.improve")
admission_module = importlib.import_module("cognee.modules.improve.admission")
users_methods = importlib.import_module("cognee.modules.users.methods")
session_manager_module = importlib.import_module(
    "cognee.infrastructure.session.get_session_manager"
)
metrics_module = importlib.import_module("cognee.modules.session_lifecycle.metrics")

USER_A = uuid4()
USER_B = uuid4()
DATASET = uuid4()


def _improve_result(*, lock_held=False, status="completed"):
    return SimpleNamespace(lock_held=lock_held, status=status)


class _Harness:
    """Patches the sweep's collaborators; ``pending`` maps session_id -> pending Q&A."""

    def __init__(self, sessions, pending, improve=None, skip_reason=None):
        self.sessions = sessions
        self.pending = pending
        self.improve = improve or AsyncMock(return_value=_improve_result())
        self.skip_reason = skip_reason

    def __enter__(self):
        async def pending_qa_count(_session_manager, _user_id, session_id):
            return self.pending.get(session_id, 0)

        self._patches = [
            patch.object(
                session_manager_module,
                "get_session_manager",
                return_value=SimpleNamespace(is_available=True),
            ),
            patch.object(
                metrics_module, "list_idle_sessions", new=AsyncMock(return_value=self.sessions)
            ),
            patch.object(consolidation, "_pending_qa_count", new=pending_qa_count),
            patch.object(
                users_methods,
                "get_user",
                new=AsyncMock(side_effect=lambda user_id: SimpleNamespace(id=user_id)),
            ),
            patch.object(
                admission_module,
                "auto_improve_skip_reason",
                new=AsyncMock(return_value=self.skip_reason),
            ),
            patch.object(improve_package, "improve", new=self.improve),
        ]
        for active in self._patches:
            active.start()
        return self

    def __exit__(self, *exc):
        for active in reversed(self._patches):
            active.stop()
        return False


@pytest.mark.asyncio
async def test_bridges_only_sessions_with_pending_entries():
    sessions = [(USER_A, "pending", DATASET), (USER_B, "caught_up", DATASET)]
    with _Harness(sessions, pending={"pending": 3}) as harness:
        report = await consolidation.consolidate_idle_sessions()

    harness.improve.assert_awaited_once()
    kwargs = harness.improve.await_args.kwargs
    assert kwargs["dataset"] == DATASET
    assert kwargs["session_ids"] == ["pending"]
    assert kwargs["user"].id == USER_A
    assert report.sessions_considered == 2
    assert report.consolidated == [(str(USER_A), "pending")]
    assert report.outcomes == {
        consolidation.OUTCOME_CONSOLIDATED: 1,
        consolidation.OUTCOME_NOTHING_PENDING: 1,
    }


@pytest.mark.asyncio
async def test_unattributed_session_is_never_bridged_into_a_guessed_dataset():
    with _Harness([(USER_A, "unscoped", None)], pending={"unscoped": 1}) as harness:
        report = await consolidation.consolidate_idle_sessions()

    harness.improve.assert_not_awaited()
    assert report.outcomes == {consolidation.OUTCOME_UNATTRIBUTED: 1}


@pytest.mark.asyncio
async def test_admission_check_can_decline():
    with _Harness(
        [(USER_A, "s1", DATASET)], pending={"s1": 2}, skip_reason="insufficient_credits"
    ) as harness:
        report = await consolidation.consolidate_idle_sessions()

    harness.improve.assert_not_awaited()
    assert report.outcomes == {consolidation.OUTCOME_ADMISSION_DECLINED: 1}


@pytest.mark.asyncio
async def test_lock_held_and_errored_results_are_not_counted_as_consolidated():
    improve = AsyncMock(
        side_effect=[_improve_result(lock_held=True), _improve_result(status="errored")]
    )
    sessions = [(USER_A, "locked", DATASET), (USER_A, "failed", DATASET)]
    with _Harness(sessions, pending={"locked": 1, "failed": 1}, improve=improve):
        report = await consolidation.consolidate_idle_sessions()

    assert report.consolidated == []
    assert report.outcomes == {
        consolidation.OUTCOME_LOCK_HELD: 1,
        consolidation.OUTCOME_ERRORED: 1,
    }


@pytest.mark.asyncio
async def test_one_failing_session_does_not_stop_the_sweep():
    improve = AsyncMock(side_effect=[RuntimeError("llm down"), _improve_result()])
    sessions = [(USER_A, "boom", DATASET), (USER_B, "ok", DATASET)]
    with _Harness(sessions, pending={"boom": 1, "ok": 1}, improve=improve):
        report = await consolidation.consolidate_idle_sessions()

    assert report.consolidated == [(str(USER_B), "ok")]
    assert report.outcomes == {
        consolidation.OUTCOME_ERRORED: 1,
        consolidation.OUTCOME_CONSOLIDATED: 1,
    }


@pytest.mark.asyncio
async def test_sweep_is_a_noop_without_a_session_cache():
    listing = AsyncMock()
    with (
        patch.object(
            session_manager_module,
            "get_session_manager",
            return_value=SimpleNamespace(is_available=False),
        ),
        patch.object(metrics_module, "list_idle_sessions", new=listing),
    ):
        report = await consolidation.consolidate_idle_sessions()

    assert report.sessions_considered == 0
    listing.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_window_uses_idle_threshold_and_cache_ttl():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    listing = AsyncMock(return_value=[])
    with (
        patch.object(
            session_manager_module,
            "get_session_manager",
            return_value=SimpleNamespace(is_available=True),
        ),
        patch.object(metrics_module, "list_idle_sessions", new=listing),
        patch.object(
            consolidation,
            "get_improve_config",
            return_value=ImproveConfig(
                idle_consolidation_after_seconds=600, idle_consolidation_batch_size=7
            ),
        ),
        patch(
            "cognee.infrastructure.databases.cache.config.get_cache_config",
            return_value=SimpleNamespace(session_ttl_seconds=3600),
        ),
    ):
        await consolidation.consolidate_idle_sessions(now=now)

    assert listing.await_args.kwargs == {
        "idle_before": now - timedelta(seconds=600),
        "active_after": now - timedelta(seconds=3600),
        "limit": 7,
    }


@pytest.mark.asyncio
async def test_pending_count_is_entries_past_the_persist_watermark():
    from cognee.infrastructure.session.session_persist_watermark import save_persisted_qa_count

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ),
    ):
        from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter
        from cognee.infrastructure.session.session_manager import SessionManager

        adapter = FSCacheAdapter()
        session_manager = SessionManager(adapter)
        user_id = str(uuid4())
        try:
            for index in range(3):
                await adapter.create_qa_entry(
                    user_id,
                    "s1",
                    question=f"q{index}",
                    context="context",
                    answer=f"a{index}",
                    qa_id=f"qa_{index}",
                )
            assert await consolidation._pending_qa_count(session_manager, user_id, "s1") == 3

            await save_persisted_qa_count(session_manager, user_id, "s1", 1)
            assert await consolidation._pending_qa_count(session_manager, user_id, "s1") == 2
        finally:
            adapter.cache.close()


@pytest.mark.asyncio
async def test_list_idle_sessions_filters_by_idle_window_and_ttl():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.metrics import list_idle_sessions
    from cognee.modules.session_lifecycle.models import SessionRecord

    now = datetime.now(timezone.utc)
    user_id = uuid4()
    rows = {
        "idle_older": now - timedelta(hours=5),
        "idle_newer": now - timedelta(hours=2),
        "active": now - timedelta(minutes=5),
        "expired": now - timedelta(days=30),
    }

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    async with engine.get_async_session() as session:
        for session_id, last_activity in rows.items():
            session.add(
                SessionRecord(
                    session_id=f"{session_id}_{user_id}",
                    user_id=user_id,
                    dataset_id=None,
                    status="running",
                    started_at=last_activity,
                    last_activity_at=last_activity,
                )
            )
        await session.commit()

    try:
        listed = await list_idle_sessions(
            idle_before=now - timedelta(hours=1),
            active_after=now - timedelta(days=7),
            limit=1000,
        )
        mine = [session_id for row_user, session_id, _ in listed if row_user == user_id]
        assert mine == [f"idle_older_{user_id}", f"idle_newer_{user_id}"]
    finally:
        async with engine.get_async_session() as session:
            for session_id in rows:
                row = await session.get(SessionRecord, (f"{session_id}_{user_id}", user_id))
                if row:
                    await session.delete(row)
            await session.commit()


@pytest.mark.asyncio
async def test_loop_sweeps_on_interval_and_stops_on_event():
    stop = asyncio.Event()
    sweeps = 0

    async def sweep():
        nonlocal sweeps
        sweeps += 1
        if sweeps == 2:
            stop.set()

    with (
        patch.object(
            consolidation,
            "get_improve_config",
            return_value=ImproveConfig(idle_consolidation_interval_seconds=0.01),
        ),
        patch.object(consolidation, "consolidate_idle_sessions", new=sweep),
    ):
        await asyncio.wait_for(consolidation.run_idle_consolidation_loop(stop), timeout=2)

    assert sweeps == 2


@pytest.mark.asyncio
async def test_loop_survives_a_failing_sweep():
    stop = asyncio.Event()
    calls = 0

    async def sweep():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("db down")
        stop.set()

    with (
        patch.object(
            consolidation,
            "get_improve_config",
            return_value=ImproveConfig(idle_consolidation_interval_seconds=0.01),
        ),
        patch.object(consolidation, "consolidate_idle_sessions", new=sweep),
    ):
        await asyncio.wait_for(consolidation.run_idle_consolidation_loop(stop), timeout=2)

    assert calls == 2


def test_consolidation_is_off_by_default():
    assert ImproveConfig().idle_consolidation_enabled is False


@pytest.mark.parametrize(
    "field_name",
    [
        "idle_consolidation_after_seconds",
        "idle_consolidation_interval_seconds",
        "idle_consolidation_batch_size",
    ],
)
def test_non_positive_consolidation_settings_are_rejected(field_name):
    with pytest.raises(ValidationError, match=field_name):
        ImproveConfig(**{field_name: 0})
