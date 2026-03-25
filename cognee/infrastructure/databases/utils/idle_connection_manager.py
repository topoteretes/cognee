"""Background manager that closes idle database connections after a timeout."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_SEC = 300  # 5 minutes
DEFAULT_CHECK_INTERVAL_SEC = 60  # check every minute


class IdleConnectionManager:
    """Tracks usage of a database wrapper and closes it after idle timeout.

    The wrapper must support a ``close()`` method. It is the wrapper's
    responsibility to restore its own connection when used again after
    being closed (e.g. lazily in its query method).

    Usage::

        manager = IdleConnectionManager(wrapper, idle_timeout_sec=300)

        # Guard every database operation:
        async with manager.using():
            await do_work()

        # When no operations run for idle_timeout_sec, wrapper.close()
        # is called automatically by the background task.

        # To tear down:
        manager.stop()
    """

    def __init__(
        self,
        wrapper,
        idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC,
        check_interval_sec: float = DEFAULT_CHECK_INTERVAL_SEC,
    ):
        self._wrapper = wrapper
        self._idle_timeout_sec = idle_timeout_sec
        self._check_interval_sec = check_interval_sec

        self._active_count: int = 0
        self._last_used: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stopped = False

    def _ensure_started(self):
        """Start the background check task if not already running."""
        if self._task is None or self._task.done():
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._check_loop())
            except RuntimeError:
                pass

    async def start_using(self):
        """Mark the start of a database operation.

        Prevents the background task from closing the wrapper while in use.
        """
        async with self._lock:
            self._active_count += 1
            self._ensure_started()

    async def end_using(self):
        """Mark the end of a database operation.

        Records the timestamp when the last concurrent user finishes.
        """
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            if self._active_count == 0:
                self._last_used = time.monotonic()

    def using(self) -> "_UsageGuard":
        """Return an async context manager that guards a database operation.

        Prevents the background task from closing the wrapper while in use.

        Usage::

            async with manager.using():
                await wrapper.query(...)
        """
        return _UsageGuard(self)

    async def _check_loop(self):
        """Periodically check for idle connections and close them."""
        while not self._stopped:
            await asyncio.sleep(self._check_interval_sec)

            async with self._lock:
                if self._active_count > 0:
                    continue
                elapsed = time.monotonic() - self._last_used
                if elapsed < self._idle_timeout_sec:
                    continue

                logger.info(
                    "Closing idle connection for %s (idle %.0fs)",
                    type(self._wrapper).__name__,
                    elapsed,
                )
                close = self._wrapper.close()
                if asyncio.iscoroutine(close):
                    await close

    def stop(self):
        """Stop the background check task."""
        self._stopped = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None


class _UsageGuard:
    """Async context manager returned by ``IdleConnectionManager.using()``."""

    __slots__ = ("_manager",)

    def __init__(self, manager: IdleConnectionManager):
        self._manager = manager

    async def __aenter__(self):
        await self._manager.start_using()
        return self._manager._wrapper

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._manager.end_using()
        return False
