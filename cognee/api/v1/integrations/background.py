"""Detached integration work that the server's shutdown drain still observes."""

import asyncio
import logging

from cognee.infrastructure.background_tasks import register_background_task

logger = logging.getLogger(__name__)

# Detached post-install / webhook work (initial syncs, event handling) runs
# off the request path; strong references keep the tasks alive until done,
# the same pattern as remember()'s _BACKGROUND_REMEMBER_TASKS.
_BACKGROUND_INTEGRATION_TASKS: set = set()


def spawn_background(coro, *, description: str) -> None:
    """Run ``coro`` detached, logging (never raising) on failure."""

    async def _guarded():
        try:
            await coro
        except Exception:  # detached work must log, not crash the loop
            logger.exception("%s failed", description)

    task = register_background_task(asyncio.create_task(_guarded()))
    _BACKGROUND_INTEGRATION_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_INTEGRATION_TASKS.discard)
