"""Semaphore-backed dataset queue.

This module exposes a lightweight, in-process concurrency limiter used to
cap how many dataset-level operations (``search`` and
``run_pipeline_per_dataset``) may execute at the same time.

The queue is intentionally simple: it is an ``asyncio.Semaphore`` wrapped in
a small object that provides an ``acquire()`` async context manager, an
``execute()`` helper, and an ``acquire_with_timeout()`` coroutine. When the
queue is disabled by configuration it becomes a no-op — ``acquire()`` yields
immediately and ``execute()`` calls the supplied function directly.

Public surface:
    - :class:`DatasetQueue`
    - :class:`DatasetQueueTimeoutError`
    - :func:`dataset_queue` — the process-wide singleton accessor.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

from cognee.shared.logging_utils import get_logger

from . import config as _config_module


logger = get_logger("cognee.dataset_queue")


T = TypeVar("T")


class DatasetQueueTimeoutError(Exception):
    """Raised when the queue cannot acquire a slot within the given timeout."""


class DatasetQueue:
    """Concurrency limiter for dataset-level operations.

    The queue is created from a :class:`DatasetQueueConfig` instance. When the
    configuration has ``dataset_queue_enabled`` set to ``False`` the queue
    becomes a no-op, meaning ``acquire()`` and ``execute()`` do not block and
    impose no concurrency limit.
    """

    def __init__(self, config: Any) -> None:
        enabled = bool(getattr(config, "dataset_queue_enabled", False))
        max_concurrent = int(getattr(config, "database_max_lru_cache_size", 1) or 1)
        if max_concurrent < 1:
            max_concurrent = 1

        self._enabled: bool = enabled
        self._max_concurrent: int = max_concurrent
        # Semaphores created outside of a running loop are still valid on
        # Python 3.10+ — the loop is bound lazily on first acquire/release.
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)

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
        """Number of slots currently available.

        Returns the semaphore's internal counter. When the queue is disabled
        the counter still reflects ``max_concurrent`` because no acquisitions
        occur.
        """
        return self._semaphore._value  # type: ignore[attr-defined]

    # ------------------------------------------------------------ acquiring
    @asynccontextmanager
    async def acquire(self):
        """Async context manager that holds a queue slot for its body.

        When the queue is disabled the context manager yields immediately
        without touching the semaphore. The slot is always released on exit,
        including on exception propagation or task cancellation.
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

        This primitive is primarily useful for monitoring and health probing
        — it acquires without an accompanying release, so callers that use
        it must release the slot manually via :attr:`_semaphore.release`.
        When the queue is disabled this is a no-op.

        Raises:
            DatasetQueueTimeoutError: if no slot becomes available before the
                timeout expires.
        """
        if not self._enabled:
            return

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:  # pragma: no cover - trivial
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
        returns a value — all three shapes are supported.

        Args:
            func: Callable (sync or async) to execute through the queue.
            *args: Positional arguments forwarded to ``func``.
            timeout: Optional timeout (seconds) for acquiring a slot. If the
                queue is disabled or ``None``, no timeout is applied to
                acquisition. The timeout does NOT apply to ``func`` itself.
            **kwargs: Keyword arguments forwarded to ``func``.

        Raises:
            DatasetQueueTimeoutError: if ``timeout`` elapses before a slot
                becomes available.
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

    The singleton is lazily constructed on first access using the current
    :func:`get_dataset_queue_config`. Tests reset the singleton by assigning
    ``dataset_queue._instance = None``; subsequent calls will pick up the
    current (possibly patched) configuration.
    """
    if dataset_queue._instance is not None:  # type: ignore[attr-defined]
        return dataset_queue._instance  # type: ignore[attr-defined]

    # Access the config via the module so ``mock.patch`` on
    # ``cognee.infrastructure.databases.dataset_queue.config.get_dataset_queue_config``
    # takes effect. A module-level ``from .config import ...`` would bind the
    # original function at import time and defeat the patch.
    cfg = _config_module.get_dataset_queue_config()
    instance = DatasetQueue(cfg)
    dataset_queue._instance = instance  # type: ignore[attr-defined]
    return instance


# Singleton storage — reset between tests by the reset_queue_singleton
# fixture. Not thread-safe, but safe within a single asyncio event loop.
dataset_queue._instance = None  # type: ignore[attr-defined]
