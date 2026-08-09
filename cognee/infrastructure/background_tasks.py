"""Registry for fire-and-forget background tasks (auto-improve, background remember).

asyncio only keeps weak references to tasks: a fire-and-forget task whose creator
drops the returned handle can be garbage-collected mid-flight, and a process that
exits while tasks are pending kills them silently ("Task was destroyed but it is
pending"). Every background task cognee spawns goes through this registry so it is

- anchored against GC for its whole lifetime, and
- drainable at shutdown (FastAPI lifespan) or explicitly via
  ``cognee.wait_for_background_tasks()`` from scripts that would otherwise exit
  before their auto-improve pass finishes.
"""

import asyncio
from typing import Coroutine, Optional, Set

from cognee.shared.logging_utils import get_logger

logger = get_logger("background_tasks")

_background_tasks: Set[asyncio.Task] = set()


def spawn_background_task(coroutine: Coroutine, *, name: Optional[str] = None) -> asyncio.Task:
    """Create an asyncio task anchored in the registry until it finishes."""
    task = asyncio.create_task(coroutine, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def pending_background_task_count() -> int:
    """Number of registered tasks that have not finished yet."""
    return sum(1 for task in _background_tasks if not task.done())


async def wait_for_background_tasks(timeout: Optional[float] = None) -> bool:
    """Wait for every registered background task to finish.

    Returns True when everything drained, False on timeout (remaining tasks keep
    running — nothing is cancelled). Task exceptions are logged, never re-raised:
    background work is best-effort by definition.
    """
    pending = [task for task in _background_tasks if not task.done()]
    if not pending:
        return True

    logger.info("Draining %d background task(s)...", len(pending))
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in done:
        if not task.cancelled() and task.exception() is not None:
            logger.warning("Background task %r failed: %s", task.get_name(), task.exception())
    if still_pending:
        logger.warning(
            "%d background task(s) still pending after drain timeout", len(still_pending)
        )
        return False
    return True
