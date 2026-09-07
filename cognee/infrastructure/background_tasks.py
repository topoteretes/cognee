"""Process-wide registry of fire-and-forget background tasks.

``remember(run_in_background=True)`` and the session-to-graph bridge that
``remember(session_id=...)`` launches run as detached ``asyncio`` tasks. The
event loop keeps only a weak reference to a task, so every launcher already
anchors its tasks in a module-level set (#4312). This module adds the one
thing those private sets cannot offer: a single place to wait for all of them.

* :func:`register_background_task` anchors a task here and drops it on
  completion. Launchers call it in addition to their own set.
* :func:`wait_for_background_tasks` awaits every registered task on the
  current event loop, with an optional timeout. The FastAPI lifespan calls it
  on shutdown so an in-flight improve is not cut off by process exit, and it
  is exported as ``cognee.wait_for_background_tasks()`` for scripts that want
  to finish their work before the interpreter goes away.

Nothing here cancels a task: a timeout reports ``False`` and leaves the work
running.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Set

from cognee.shared.logging_utils import get_logger

logger = get_logger("background_tasks")

_BACKGROUND_TASKS: Set["asyncio.Task"] = set()


def register_background_task(task: "asyncio.Task") -> "asyncio.Task":
    """Anchor ``task`` until it finishes and return it unchanged."""
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def _pending_on_current_loop() -> Set["asyncio.Task"]:
    """Registered tasks that are still running on the calling event loop.

    Tasks from another loop (a previous ``asyncio.run``) cannot be awaited
    from here and are left to their own loop.
    """
    loop = asyncio.get_running_loop()
    pending: Set["asyncio.Task"] = set()
    for task in list(_BACKGROUND_TASKS):
        if task.done():
            _BACKGROUND_TASKS.discard(task)
            continue
        try:
            if task.get_loop() is not loop:
                continue
        except RuntimeError:
            continue
        pending.add(task)
    return pending


def pending_background_tasks() -> int:
    """Number of registered tasks that have not finished yet."""
    return sum(1 for task in _BACKGROUND_TASKS if not task.done())


async def wait_for_background_tasks(timeout: Optional[float] = None) -> bool:
    """Wait until every registered background task has finished.

    Tasks may launch further background tasks while draining (a background
    remember that fires an improve), so the wait loops until the registry is
    empty. Task failures are the launcher's business — each background body
    records its own error — so they are not re-raised here.

    Returns ``True`` when everything finished, ``False`` when ``timeout``
    seconds elapsed with work still pending. Nothing is cancelled either way.
    """
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

    while True:
        pending = _pending_on_current_loop()
        if not pending:
            return True

        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            logger.warning(
                "wait_for_background_tasks: %d task(s) still running after %.1fs",
                len(pending),
                timeout or 0.0,
            )
            return False

        done, _ = await asyncio.wait(pending, timeout=remaining)
        for task in done:
            # Consume the result so a failed task does not log "exception was
            # never retrieved" — the launcher already recorded the error.
            if not task.cancelled():
                task.exception()
        # A timeout fall-through is reported by the deadline check above.
