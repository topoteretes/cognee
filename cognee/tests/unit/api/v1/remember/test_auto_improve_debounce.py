"""Auto-improve debounce and the background-task registry."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.remember.remember import AUTO_IMPROVE_STATE_ID, _should_auto_improve
from cognee.infrastructure.background_tasks import (
    spawn_background_task,
    wait_for_background_tasks,
)

remember_module = sys.modules["cognee.api.v1.remember.remember"]


class _StateStore:
    def __init__(self):
        self.rows = {}

    def session_manager(self):
        sm = SimpleNamespace(is_available=True)
        sm.get_session_context_entries = AsyncMock(
            side_effect=lambda **_kw: list(self.rows.values())
        )

        async def update(*, user_id, session_id, entry_id, merge):
            if entry_id in self.rows:
                self.rows[entry_id] = {**self.rows[entry_id], **merge}
                return True
            return False

        async def create(*, user_id, session_id, entry_dump):
            self.rows[entry_dump["id"]] = dict(entry_dump)

        sm.update_session_context_entry = AsyncMock(side_effect=update)
        sm.create_session_context_entry = AsyncMock(side_effect=create)
        return sm


def _user():
    return SimpleNamespace(id=uuid4())


@pytest.mark.asyncio
async def test_auto_improve_debounces_until_entry_threshold(monkeypatch):
    monkeypatch.setattr(remember_module, "AUTO_IMPROVE_MIN_ENTRIES", 3)
    monkeypatch.setattr(remember_module, "AUTO_IMPROVE_MIN_INTERVAL_SECONDS", 3600.0)
    store = _StateStore()
    user = _user()

    with patch(
        "cognee.infrastructure.session.get_session_manager.get_session_manager",
        side_effect=store.session_manager,
    ):
        # First call: no state yet -> fires (and stamps last_improve_at).
        assert await _should_auto_improve(user, "s1") is True
        # Two quick follow-ups stay under both thresholds.
        assert await _should_auto_improve(user, "s1") is False
        assert await _should_auto_improve(user, "s1") is False
        # Third new entry hits AUTO_IMPROVE_MIN_ENTRIES.
        assert await _should_auto_improve(user, "s1") is True
        # Counter reset after firing.
        assert store.rows[AUTO_IMPROVE_STATE_ID]["entries_since_improve"] == 0


@pytest.mark.asyncio
async def test_auto_improve_fires_after_interval(monkeypatch):
    monkeypatch.setattr(remember_module, "AUTO_IMPROVE_MIN_ENTRIES", 100)
    monkeypatch.setattr(remember_module, "AUTO_IMPROVE_MIN_INTERVAL_SECONDS", 0.0)
    store = _StateStore()
    user = _user()

    with patch(
        "cognee.infrastructure.session.get_session_manager.get_session_manager",
        side_effect=store.session_manager,
    ):
        assert await _should_auto_improve(user, "s1") is True
        # Interval 0 means every call is due regardless of the entry counter.
        assert await _should_auto_improve(user, "s1") is True


@pytest.mark.asyncio
async def test_auto_improve_fails_open_when_cache_breaks():
    sm = SimpleNamespace(is_available=True)
    sm.get_session_context_entries = AsyncMock(side_effect=RuntimeError("cache down"))

    with patch(
        "cognee.infrastructure.session.get_session_manager.get_session_manager",
        return_value=sm,
    ):
        assert await _should_auto_improve(_user(), "s1") is True


@pytest.mark.asyncio
async def test_background_task_registry_anchors_and_drains():
    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()
        return "done"

    task = spawn_background_task(work(), name="test-task")
    await started.wait()

    # Drain times out while the task is blocked, without cancelling it —
    # only possible if the registry actually anchored the task.
    assert await wait_for_background_tasks(timeout=0.01) is False
    assert not task.cancelled()

    release.set()
    assert await wait_for_background_tasks(timeout=5) is True
    assert task.result() == "done"
