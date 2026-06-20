"""Opt-in LLM call timeouts and slow-call diagnostics for local inference (#2902)."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Awaitable, TypeVar

from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class LLMCallTimeoutError(TimeoutError):
    """Raised when an opt-in LLM call timeout is exceeded."""


def build_llm_call_label(*, provider: str, model: str, endpoint: str | None) -> str:
    endpoint_display = endpoint or "<default>"
    return f"provider={provider} model={model} endpoint={endpoint_display}"


async def run_with_llm_call_limits(
    coro: Awaitable[T],
    *,
    label: str,
    timeout_seconds: int,
    slow_warning_seconds: int,
) -> T:
    """
    Execute an LLM request with optional diagnostics.

    - ``timeout_seconds=0`` disables wall-clock abort (default, unlike #3126).
    - ``slow_warning_seconds`` logs a warning if the call is still running after N seconds.
    """
    start = time.monotonic()
    logger.info("LLM call started (%s)", label)
    warn_task: asyncio.Task | None = None

    async def _log_slow_call_warning() -> None:
        await asyncio.sleep(slow_warning_seconds)
        logger.warning(
            "LLM call still running after %ss (%s). "
            "For local inference hangs, set LLM_CALL_TIMEOUT_SECONDS to fail fast.",
            slow_warning_seconds,
            label,
        )

    try:
        if slow_warning_seconds > 0:
            warn_task = asyncio.create_task(_log_slow_call_warning())

        if timeout_seconds > 0:
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError as error:
                raise LLMCallTimeoutError(
                    f"LLM call exceeded LLM_CALL_TIMEOUT_SECONDS={timeout_seconds} ({label}). "
                    "Increase the timeout or set LLM_CALL_TIMEOUT_SECONDS=0 to disable opt-in limits."
                ) from error
        else:
            result = await coro

        elapsed = time.monotonic() - start
        if slow_warning_seconds > 0 and elapsed >= slow_warning_seconds:
            logger.warning("LLM call completed slowly in %.2fs (%s)", elapsed, label)
        else:
            logger.info("LLM call finished in %.2fs (%s)", elapsed, label)
        return result
    except Exception:
        logger.error(
            "LLM call failed after %.2fs (%s)",
            time.monotonic() - start,
            label,
            exc_info=True,
        )
        raise
    finally:
        if warn_task is not None:
            warn_task.cancel()
            with suppress(asyncio.CancelledError):
                await warn_task
