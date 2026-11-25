"""Detached integration work: install hooks, webhooks, manual and scheduled syncs."""

import asyncio

from cognee.infrastructure.background_tasks import register_background_task
from cognee.shared.logging_utils import get_logger

logger = get_logger("integrations_background")

# Strong references keep the tasks alive until done, the same pattern as
# remember()'s _BACKGROUND_REMEMBER_TASKS.
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
