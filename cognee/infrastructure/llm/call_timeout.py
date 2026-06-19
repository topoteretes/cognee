import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from cognee.infrastructure.llm.exceptions import LLMCallTimeoutError

T = TypeVar("T")


def _consume_task_exception(task: asyncio.Future) -> None:
    """Retrieve a detached task's exception so asyncio does not log it as unhandled."""
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def run_with_timeout(awaitable: Awaitable[T], *, timeout_seconds: float, operation: str) -> T:
    """Run an LLM operation with a hard deadline for the awaiting caller.

    ``asyncio.wait_for`` waits for cancellation cleanup before raising. Some SDK retry
    layers delay or suppress that cleanup, allowing a nominal timeout to overrun by
    minutes. This race cancels the provider task at the deadline but does not wait for
    it to acknowledge cancellation.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait((task,), timeout=timeout_seconds)
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_consume_task_exception)
        raise

    if task in done:
        return task.result()

    task.cancel()
    task.add_done_callback(_consume_task_exception)
    raise LLMCallTimeoutError(operation=operation, timeout_seconds=timeout_seconds)
