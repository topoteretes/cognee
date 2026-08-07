"""Process-local queue lifecycle for session maintenance."""

import asyncio
from contextlib import suppress
from contextvars import Context
from dataclasses import dataclass, field

from cognee.infrastructure.session.session_search_models import SessionMaintenanceWorkItem
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_maintenance_worker")
MAX_QUEUED_MAINTENANCE = 100


@dataclass
class _WorkerState:
    queue: asyncio.Queue[SessionMaintenanceWorkItem] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_QUEUED_MAINTENANCE)
    )
    task: asyncio.Task | None = None
    # Evidence this worker owns, queued or in flight. Distillation must not touch it.
    tracked_ids: set[str] = field(default_factory=set)


_states: dict[asyncio.AbstractEventLoop, _WorkerState] = {}


async def _run_worker(state: _WorkerState) -> None:
    from cognee.infrastructure.session.session_maintenance import process_session_maintenance

    while True:
        work_item = await state.queue.get()
        try:
            await process_session_maintenance(work_item)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Session maintenance failed: %s", error)
        finally:
            state.tracked_ids.discard(work_item.evidence_id)
            state.queue.task_done()


def _start_worker(state: _WorkerState) -> None:
    if state.task is not None and not state.task.done():
        return
    coroutine = _run_worker(state)
    state.task = Context().run(asyncio.create_task, coroutine)


async def enqueue_session_maintenance(
    work_item: SessionMaintenanceWorkItem,
    session_manager,
) -> bool:
    """Queue durable evidence without waiting for its maintenance operation."""
    loop = asyncio.get_running_loop()
    state = _states.setdefault(loop, _WorkerState())
    if work_item.evidence_id in state.tracked_ids:
        return False

    try:
        state.queue.put_nowait(work_item)
    except asyncio.QueueFull:
        try:
            updated = await session_manager.update_session_context_entry(
                user_id=work_item.user_id,
                session_id=work_item.session_id,
                entry_id=work_item.evidence_id,
                merge={"status": "deferred", "error": None},
            )
            if not updated:
                logger.warning("Could not defer session evidence: %s", work_item.evidence_id)
        except Exception:
            logger.warning("Could not defer session evidence: %s", work_item.evidence_id)
        return False

    state.tracked_ids.add(work_item.evidence_id)
    _start_worker(state)
    return True


def get_tracked_evidence_ids() -> set[str]:
    """Return evidence owned by a live worker in this process.

    Distillation may run on a different loop than the worker that owns the evidence, so
    every live loop counts. A closed loop owns nothing: its evidence is abandoned and
    stays recoverable.
    """
    tracked = set()
    for loop, state in list(_states.items()):
        if loop.is_closed():
            continue
        tracked.update(state.tracked_ids)
    return tracked


async def _cancel_worker(state: _WorkerState) -> None:
    if state.task is None:
        return
    state.task.cancel()
    with suppress(asyncio.CancelledError):
        await state.task


async def drain_session_maintenance(timeout_seconds: float = 30.0) -> None:
    """Finish maintenance on this loop, or cancel it cleanly on timeout.

    Short-lived SDK callers should call this in the same coroutine as ``cognee.search``.
    Search returns before background maintenance completes.
    """
    loop = asyncio.get_running_loop()
    state = _states.get(loop)
    if state is None:
        if _states:
            raise RuntimeError("session maintenance must be drained on its originating event loop")
        return

    try:
        await asyncio.wait_for(state.queue.join(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        await _cancel_worker(state)
        while True:
            try:
                queued = state.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            state.tracked_ids.discard(queued.evidence_id)
            state.queue.task_done()
        _states.pop(loop, None)
        raise

    await _cancel_worker(state)
    _states.pop(loop, None)
