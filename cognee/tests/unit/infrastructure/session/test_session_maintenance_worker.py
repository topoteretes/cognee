import asyncio
from contextvars import ContextVar
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.session.session_search_models import SessionMaintenanceWorkItem
import cognee.infrastructure.session.session_maintenance as maintenance
import cognee.infrastructure.session.session_maintenance_worker as worker


def _work(evidence_id: str) -> SessionMaintenanceWorkItem:
    return SessionMaintenanceWorkItem(
        evidence_id=evidence_id,
        user_id="user",
        session_id="session",
    )


@pytest.mark.asyncio
async def test_enqueue_returns_before_blocked_work_and_drain_finishes(monkeypatch):
    release = asyncio.Event()
    started = asyncio.Event()

    async def blocked(_work_item):
        started.set()
        await release.wait()

    monkeypatch.setattr(maintenance, "process_session_maintenance", blocked)
    manager = AsyncMock()

    assert await worker.enqueue_session_maintenance(_work("e1"), manager)
    await asyncio.wait_for(started.wait(), timeout=1)
    assert worker.get_tracked_evidence_ids() == {"e1"}
    release.set()
    await worker.drain_session_maintenance()
    assert worker.get_tracked_evidence_ids() == set()


@pytest.mark.asyncio
async def test_worker_processes_serially_and_restarts_after_drain(monkeypatch):
    active = 0
    max_active = 0
    processed = []

    async def process(work_item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        processed.append(work_item.evidence_id)
        active -= 1

    monkeypatch.setattr(maintenance, "process_session_maintenance", process)
    manager = AsyncMock()
    await worker.enqueue_session_maintenance(_work("e1"), manager)
    await worker.enqueue_session_maintenance(_work("e2"), manager)
    await worker.drain_session_maintenance()
    await worker.enqueue_session_maintenance(_work("e3"), manager)
    await worker.drain_session_maintenance()

    assert processed == ["e1", "e2", "e3"]
    assert max_active == 1


@pytest.mark.asyncio
async def test_queue_full_marks_new_evidence_deferred(monkeypatch):
    monkeypatch.setattr(worker, "MAX_QUEUED_MAINTENANCE", 1)
    monkeypatch.setattr(worker, "_start_worker", lambda state: None)
    manager = AsyncMock()
    manager.update_session_context_entry.return_value = True

    assert await worker.enqueue_session_maintenance(_work("e1"), manager)
    assert not await worker.enqueue_session_maintenance(_work("e2"), manager)
    manager.update_session_context_entry.assert_awaited_once_with(
        user_id="user",
        session_id="session",
        entry_id="e2",
        merge={"status": "deferred", "error": None},
    )
    with pytest.raises(TimeoutError):
        await worker.drain_session_maintenance(timeout_seconds=0)


@pytest.mark.asyncio
async def test_timeout_clears_state_and_next_enqueue_restarts(monkeypatch):
    release = asyncio.Event()

    async def blocked(_work_item):
        await release.wait()

    monkeypatch.setattr(maintenance, "process_session_maintenance", blocked)
    manager = AsyncMock()
    await worker.enqueue_session_maintenance(_work("e1"), manager)
    with pytest.raises(TimeoutError):
        await worker.drain_session_maintenance(timeout_seconds=0.01)
    assert worker.get_tracked_evidence_ids() == set()

    processed = AsyncMock()
    monkeypatch.setattr(maintenance, "process_session_maintenance", processed)
    await worker.enqueue_session_maintenance(_work("e2"), manager)
    await worker.drain_session_maintenance()
    processed.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_does_not_copy_request_context(monkeypatch):
    request_value = ContextVar("request_value", default=None)
    observed = []

    async def process(_work_item):
        observed.append(request_value.get())

    monkeypatch.setattr(maintenance, "process_session_maintenance", process)
    token = request_value.set("secret request state")
    try:
        await worker.enqueue_session_maintenance(_work("e1"), AsyncMock())
        await worker.drain_session_maintenance()
    finally:
        request_value.reset(token)

    assert observed == [None]


@pytest.mark.asyncio
async def test_drain_from_another_loop_raises(monkeypatch):
    release = asyncio.Event()

    async def blocked(_work_item):
        await release.wait()

    monkeypatch.setattr(maintenance, "process_session_maintenance", blocked)
    await worker.enqueue_session_maintenance(_work("e1"), AsyncMock())

    with pytest.raises(RuntimeError, match="originating event loop"):
        await asyncio.to_thread(asyncio.run, worker.drain_session_maintenance())

    release.set()
    await worker.drain_session_maintenance()
