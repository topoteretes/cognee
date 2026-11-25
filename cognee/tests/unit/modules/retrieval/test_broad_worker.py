import asyncio
import sqlite3
import threading
from contextvars import ContextVar

import pytest

from cognee.modules.retrieval.broad_store import RecordStore
from cognee.modules.retrieval.broad_worker import BroadWorker


@pytest.mark.asyncio
async def test_worker_preserves_context_and_sqlite_thread_ownership():
    scope = ContextVar("broad_test_scope", default="unset")
    token = scope.set("request")
    threads = []

    class Store(RecordStore):
        def __init__(self):
            assert scope.get() == "request"
            threads.append(threading.get_ident())
            super().__init__()

        def close(self):
            threads.append(threading.get_ident())
            super().close()
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                self.connection.execute("SELECT 1")

    try:
        async with BroadWorker(Store) as worker:
            await worker.run(worker.store.add_document, "notes", "One\nTwo")
            assert (await worker.run(worker.store.run, "SELECT COUNT(*) FROM lines"))[1] == [(2,)]
            assert await worker.run(scope.get) == "request"
            assert await worker.run(threading.get_ident) == threads[0]
    finally:
        scope.reset(token)
    assert len(threads) == 2 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_one_busy_retrieval_does_not_queue_other_retrievals():
    started, release = threading.Event(), threading.Event()

    def busy():
        started.set()
        assert release.wait(5)

    async def slow():
        async with BroadWorker(RecordStore) as worker:
            await worker.run(busy)

    async def fast():
        async with BroadWorker(RecordStore) as worker:
            return await worker.run(worker.store.run, "SELECT 7")

    task = asyncio.create_task(slow())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert (await asyncio.wait_for(fast(), timeout=2))[1] == [(7,)]
        assert not task.done()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_cancellation_during_creation_still_closes_the_store():
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Store(RecordStore):
        def __init__(self):
            started.set()
            assert release.wait(5)
            super().__init__()

        def close(self):
            super().close()
            closed.set()

    async def retrieve():
        async with BroadWorker(Store):
            pytest.fail("A cancelled initialization entered the context")

    task = asyncio.create_task(retrieve())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not closed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_remove_queued_cleanup():
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Store(RecordStore):
        def close(self):
            super().close()
            closed.set()

    def busy():
        started.set()
        assert release.wait(5)

    async def retrieve():
        async with BroadWorker(Store) as worker:
            await worker.run(busy)

    task = asyncio.create_task(retrieve())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
    assert await asyncio.to_thread(closed.wait, 2)
