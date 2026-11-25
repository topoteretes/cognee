"""Keep one retrieval's synchronous work and SQLite connection on its own thread."""

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from typing_extensions import Self

from cognee.modules.retrieval.broad_store import RecordStore

T = TypeVar("T")


class BroadWorker:
    """Own a private store; execute its lifecycle and CPU work on one worker thread."""

    def __init__(self, factory: Callable[[], RecordStore]):
        self._factory = factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cognee-broad")
        self._store: RecordStore | None = None

    @property
    def store(self) -> RecordStore:
        if self._store is None:
            raise RuntimeError("BROAD's record store is not open")
        return self._store

    async def __aenter__(self) -> Self:
        try:
            await self.run(self._open)
        except BaseException:
            await self._aclose()
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if isinstance(exc, asyncio.CancelledError):
            # SQLite explicitly permits interrupt() from another thread. Closing still
            # waits for the running operation and happens on the connection's thread.
            self.store.connection.interrupt()
        await self._aclose()

    async def run(self, function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Run synchronous work on the store's thread with the caller's context."""
        call = partial(contextvars.copy_context().run, function, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    def _open(self) -> None:
        self._store = self._factory()

    def _close_store(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    async def _aclose(self) -> None:
        try:
            # Queue cleanup even if a previous call is still running. Shielding keeps
            # another cancellation from removing it from the executor's queue.
            closed = asyncio.wrap_future(self._executor.submit(self._close_store))
            await asyncio.shield(closed)
        finally:
            self._executor.shutdown(wait=False)
