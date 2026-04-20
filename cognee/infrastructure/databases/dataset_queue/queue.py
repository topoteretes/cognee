"""Semaphore-backed dataset queue.

This module exposes a lightweight, in-process concurrency limiter used to
cap how many dataset-level operations (``search`` fan-out and
``run_pipeline_per_dataset``) may execute at the same time.

The queue is intentionally simple: it is an ``asyncio.Semaphore`` wrapped in
a small object that provides an ``acquire()`` async context manager, an
``execute()`` helper, and an ``acquire_with_timeout()`` coroutine. When the
queue is disabled by configuration it becomes a no-op — ``acquire()`` yields
immediately and ``execute()`` calls the supplied function directly.

Configuration is read from two sources:

* ``DATASET_QUEUE_ENABLED`` — environment variable. When set to a truthy
  value (``1``, ``true``, ``yes``, ``on``; case-insensitive) the queue
  actively limits concurrency. Anything else (including unset) means off.
* ``DATABASE_MAX_LRU_CACHE_SIZE`` — the shared constant defined in
  ``cognee.shared.lru_cache``. This is the same knob that controls the
  database-adapter LRU cache sizes, so one variable governs both.

Public surface:
    - :class:`DatasetQueue`
    - :class:`DatasetQueueTimeoutError`
    - :class:`DatasetQueueSettings`
    - :func:`get_dataset_queue_settings` — the mock seam for tests.
    - :func:`dataset_queue` — the process-wide singleton accessor.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

from cognee.shared.logging_utils import get_logger
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE


logger = get_logger("cognee.dataset_queue")


T = TypeVar("T")


# Recognised truthy values for ``DATASET_QUEUE_ENABLED``. Anything else
# (including unset or empty) leaves the queue disabled.
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})


class DatasetQueueTimeoutError(Exception):
    """Raised when the queue cannot acquire a slot within the given timeout."""


@dataclass
class DatasetQueueSettings:
    """Effective runtime settings for the dataset queue.

    Attributes:
        enabled: Whether the queue should limit concurrency.
        max_concurrent: Maximum concurrent dataset operations when enabled.
    """

    enabled: bool
    max_concurrent: int


def get_dataset_queue_settings() -> DatasetQueueSettings:
    """Return the effective settings for the dataset queue.

    ``DATASET_QUEUE_ENABLED`` is consulted on every call so toggling the flag
    and resetting the singleton picks up the new state. ``max_concurrent``
    is sourced from :data:`cognee.shared.lru_cache.DATABASE_MAX_LRU_CACHE_SIZE`
    — by design this reuses the same knob as the DB-adapter LRU caches.
    """
    raw = os.getenv("DATASET_QUEUE_ENABLED", "").strip().lower()
    enabled = raw in _TRUTHY_VALUES

    max_concurrent = int(DATABASE_MAX_LRU_CACHE_SIZE or 1)
    if max_concurrent < 1:
        max_concurrent = 1

    return DatasetQueueSettings(enabled=enabled, max_concurrent=max_concurrent)


class DatasetQueue:
    """Concurrency limiter for dataset-level operations.

    When ``enabled`` is ``False`` the queue becomes a no-op: :meth:`acquire`
    and :meth:`execute` do not block and impose no concurrency limit.
    """

    def __init__(self, enabled: bool, max_concurrent: int) -> None:
        safe_max = int(max_concurrent or 1)
        if safe_max < 1:
            safe_max = 1

        self._enabled: bool = bool(enabled)
        self._max_concurrent: int = safe_max
        # Semaphores built outside a running loop are fine on Python 3.10+;
        # the loop is bound lazily on the first acquire/release.
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(safe_max)

    # ---------------------------------------------------------------- state
    @property
    def enabled(self) -> bool:
        """Whether the queue is actively limiting concurrency."""
        return self._enabled

    @property
    def max_concurrent(self) -> int:
        """Maximum number of concurrent operations permitted."""
        return self._max_concurrent

    @property
    def available_slots(self) -> int:
        """Current free-slot count (directly from the underlying semaphore)."""
        return self._semaphore._value  # type: ignore[attr-defined]

    # ------------------------------------------------------------ acquiring
    @asynccontextmanager
    async def acquire(self):
        """Async context manager holding one queue slot for its body.

        When the queue is disabled this yields immediately without touching
        the semaphore. The slot is always released on exit — on normal
        return, exception propagation, or task cancellation.
        """
        if not self._enabled:
            yield
            return

        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    async def acquire_with_timeout(self, timeout: float) -> None:
        """Attempt to acquire a slot, raising on timeout.

        Does NOT pair an automatic release — callers using this primitive are
        responsible for releasing the slot themselves. When the queue is
        disabled this is a no-op.

        Raises:
            DatasetQueueTimeoutError: if no slot becomes available before the
                timeout expires.
        """
        if not self._enabled:
            return

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise DatasetQueueTimeoutError(
                f"Could not acquire dataset queue slot within {timeout} seconds "
                f"(max_concurrent={self._max_concurrent})."
            ) from exc

    # ---------------------------------------------------------------- exec
    async def execute(
        self,
        func: Callable[..., Union[Awaitable[T], T]],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Run ``func`` through the queue, honouring the concurrency limit.

        ``func`` may be a coroutine function, a regular function that returns
        an awaitable (e.g., a ``lambda`` wrapper), or a plain function that
        returns a value.

        ``timeout`` gates the **acquire** step only — it does not cap the
        runtime of ``func`` itself.
        """

        async def _run() -> T:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result  # type: ignore[no-any-return]
            return result  # type: ignore[return-value]

        if not self._enabled:
            return await _run()

        if timeout is None:
            async with self.acquire():
                return await _run()

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise DatasetQueueTimeoutError(
                f"Could not acquire dataset queue slot within {timeout} seconds "
                f"(max_concurrent={self._max_concurrent})."
            ) from exc
        try:
            return await _run()
        finally:
            self._semaphore.release()


def dataset_queue() -> DatasetQueue:
    """Return the process-wide :class:`DatasetQueue` singleton.

    The singleton is lazily constructed on first access from the current
    :func:`get_dataset_queue_settings`. Tests reset the singleton by
    assigning ``dataset_queue._instance = None``; subsequent calls pick up
    whatever settings are in effect then (respecting ``mock.patch`` of
    :func:`get_dataset_queue_settings`).
    """
    if dataset_queue._instance is not None:  # type: ignore[attr-defined]
        return dataset_queue._instance  # type: ignore[attr-defined]

    settings = get_dataset_queue_settings()
    instance = DatasetQueue(
        enabled=settings.enabled,
        max_concurrent=settings.max_concurrent,
    )
    dataset_queue._instance = instance  # type: ignore[attr-defined]
    return instance


# Singleton storage — reset between tests by the reset_queue_singleton
# fixture. Not thread-safe, but safe within a single asyncio event loop.
dataset_queue._instance = None  # type: ignore[attr-defined]
