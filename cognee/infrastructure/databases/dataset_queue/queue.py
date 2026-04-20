"""Semaphore-backed dataset queue.

A lightweight, in-process concurrency limiter used to cap how many
dataset-level operations (``search`` fan-out and
``run_pipeline_per_dataset``) may execute at the same time.

Configuration:

* ``DATASET_QUEUE_ENABLED`` — environment variable. Truthy values
  (``1``/``true``/``yes``/``on``/``y``/``t``, case-insensitive) turn the
  queue on. Anything else (including unset) leaves it disabled.
* ``DATABASE_MAX_LRU_CACHE_SIZE`` — the shared constant defined in
  :mod:`cognee.shared.lru_cache` (default: ``128``). Sets the slot count
  when the queue is enabled.

Public surface:
    - :func:`dataset_queue` — process-wide singleton accessor.
    - :func:`get_dataset_queue_settings` — test mock seam.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from cognee.shared.logging_utils import get_logger
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE

logger = get_logger("cognee.dataset_queue")


@dataclass
class DatasetQueueSettings:
    """Effective runtime settings for the dataset queue."""

    enabled: bool
    max_concurrent: int


def get_dataset_queue_settings() -> DatasetQueueSettings:
    """Return the effective settings for the dataset queue.

    Re-reads ``DATASET_QUEUE_ENABLED`` on every call. ``max_concurrent`` is
    sourced from :data:`cognee.shared.lru_cache.DATABASE_MAX_LRU_CACHE_SIZE`
    so one knob governs both the DB-adapter LRU caches and the queue.

    Tests typically patch this function rather than twiddle the environment.
    """
    raw = os.getenv("DATASET_QUEUE_ENABLED", "").strip().lower()
    # Recognised true values for ``DATASET_QUEUE_ENABLED``.
    true_values = frozenset({"1", "true", "yes", "on", "y"})
    enabled = raw in true_values

    max_concurrent = DATABASE_MAX_LRU_CACHE_SIZE

    return DatasetQueueSettings(enabled=enabled, max_concurrent=max_concurrent)


class DatasetQueue:
    """Concurrency limiter for dataset-level operations.

    When ``enabled`` is ``False`` the queue is a no-op: :meth:`acquire` does
    not block and does not touch its internal semaphore.
    """

    def __init__(self, enabled: bool, max_concurrent: int) -> None:
        safe_max = int(max_concurrent)
        if safe_max < 1:
            self._enabled: bool = False
            self._max_concurrent: int = safe_max
            logger.debug(f"DatasetQueue disabled due to non-positive max_concurrent: {safe_max}")
        else:
            self._enabled: bool = bool(enabled)
            self._max_concurrent: int = safe_max
            self._semaphore: asyncio.Semaphore = asyncio.Semaphore(safe_max)

    @asynccontextmanager
    async def acquire(self):
        """Async context manager holding one queue slot for its body.

        No-op when the queue is disabled. The slot is released on exit —
        normal return, exception propagation, or task cancellation.
        """
        if not self._enabled:
            yield
            return

        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


def dataset_queue() -> DatasetQueue:
    """Return the process-wide :class:`DatasetQueue` singleton.

    Lazily constructed on first access from :func:`get_dataset_queue_settings`.
    Tests reset it by assigning ``dataset_queue._instance = None``.
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
dataset_queue._instance = None  # type: ignore[attr-defined]
