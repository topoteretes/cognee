"""Concurrent-RPC tests for ``SubprocessSession``.

Exercise the request-id + reader-thread design: many in-flight calls,
out-of-order completion, crash-with-N-pending, sync/async interleaving,
caller cancellation, and per-call timeout cleanup.

Uses a small async-capable worker so the worker-side concurrent dispatch
(``asyncio.create_task`` for async handlers) is also exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing as mp
import time

import pytest

from cognee_db_workers.harness import (
    Request,
    SubprocessSession,
    SubprocessTransportError,
    run_worker_loop,
    spawn_without_main,
)


# --- helper worker --------------------------------------------------------

OP_ECHO_FAST = 10
OP_ASYNC_SLEEP = 11  # awaits asyncio.sleep(arg)
OP_SYNC_SLEEP = 12  # time.sleep(arg) — exercises the sync inline branch


def _echo_fast(_registry, req):
    return req.args[0]


async def _async_sleep(_registry, req):
    await asyncio.sleep(req.args[0])
    return req.args[0]


def _sync_sleep(_registry, req):
    time.sleep(req.args[0])
    return req.args[0]


DISPATCH = {
    OP_ECHO_FAST: _echo_fast,
    OP_ASYNC_SLEEP: _async_sleep,
    OP_SYNC_SLEEP: _sync_sleep,
}


def _worker_main(req_q, resp_q):
    run_worker_loop(DISPATCH, req_q, resp_q)


def _start_session(**kwargs) -> SubprocessSession:
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    resp_q = ctx.Queue()
    proc = ctx.Process(target=_worker_main, args=(req_q, resp_q), daemon=True)
    with spawn_without_main():
        proc.start()
    session = SubprocessSession(proc, req_q, resp_q, **kwargs)
    session.wait_for_ready()
    return session


# --- tests ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_async_calls_complete_independently():
    """N concurrent ``call_async`` complete; each gets its own response.

    Each request gets a unique payload routed back via its ``request_id``
    — if the router were broken, payloads would scramble.
    """
    session = _start_session()
    try:
        n = 25
        payloads = [f"payload-{i}" for i in range(n)]
        results = await asyncio.gather(
            *(session.call_async(Request(op=OP_ECHO_FAST, args=(p,))) for p in payloads)
        )
        assert [r.result for r in results] == payloads
        # Registry is empty after every call returns.
        assert session._pending == {}
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_slow_call_does_not_block_fast_calls():
    """One slow async op runs alongside many fast ones.

    With the old single-lock design the fast ops would queue behind the
    slow one. With concurrent dispatch the fast ops finish well before
    the slow one (and well below the slow op's own duration).
    """
    session = _start_session()
    try:
        slow = asyncio.create_task(session.call_async(Request(op=OP_ASYNC_SLEEP, args=(0.6,))))
        # Give the slow request time to land on the worker.
        await asyncio.sleep(0.05)
        t0 = time.monotonic()
        fast = await asyncio.gather(
            *(session.call_async(Request(op=OP_ECHO_FAST, args=(i,))) for i in range(10))
        )
        fast_elapsed = time.monotonic() - t0
        # Fast batch should complete well before the slow op's 0.6s.
        assert fast_elapsed < 0.4, f"fast calls blocked by slow op: {fast_elapsed:.3f}s"
        assert [r.result for r in fast] == list(range(10))
        slow_resp = await slow
        assert slow_resp.result == 0.6
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_worker_crash_fails_all_pending():
    """When the worker dies with N pending calls, every awaiter raises
    ``SubprocessTransportError`` and ``_pending`` is empty afterward.
    """
    session = _start_session()
    try:
        # Submit N async-sleeping calls that will be in flight on the worker.
        tasks = [
            asyncio.create_task(session.call_async(Request(op=OP_ASYNC_SLEEP, args=(2.0,))))
            for _ in range(8)
        ]
        # Wait for them to be pending on the session.
        deadline = time.monotonic() + 2.0
        while len(session._pending) < 8 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert len(session._pending) == 8

        session._proc.kill()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, SubprocessTransportError) for r in results), results
        # The reader's finally cleared the registry as the load-bearing
        # invariant demands.
        assert session._pending == {}
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_caller_cancellation_pops_pending_entry():
    """Cancelling the awaiting task must pop the registry entry even
    though no response ever arrives.
    """
    session = _start_session()
    try:
        task = asyncio.create_task(session.call_async(Request(op=OP_ASYNC_SLEEP, args=(2.0,))))
        # Wait until it's pending on the session.
        deadline = time.monotonic() + 1.0
        while not session._pending and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert session._pending, "request never reached _pending"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Entry should be gone (try/finally pop in _issue_async).
        assert session._pending == {}
        # Session still usable for fresh calls.
        resp = await session.call_async(Request(op=OP_ECHO_FAST, args=("hi",)))
        assert resp.result == "hi"
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_per_call_timeout_cleans_registry_session_stays_open():
    """A per-call timeout pops the registry entry and leaves the session
    usable — contrast with the old design which marked the whole session
    closed on any timeout.
    """
    session = _start_session()
    try:
        with pytest.raises(TimeoutError):
            await session.call_async(Request(op=OP_ASYNC_SLEEP, args=(2.0,)), timeout=0.3)
        assert session._pending == {}
        assert not session._closed
        # Same session still works.
        resp = await session.call_async(Request(op=OP_ECHO_FAST, args=("ok",)))
        assert resp.result == "ok"
    finally:
        session.shutdown()


def test_sync_and_async_calls_interleave_on_one_session():
    """Sync ``call`` and async ``call_async`` share the same registry +
    reader; interleaving them must not corrupt responses.
    """
    import threading

    session = _start_session()
    try:
        n = 20
        sync_results: list = []
        async_results: list = []
        errors: list = []

        def do_sync():
            try:
                for i in range(n):
                    r = session.call(Request(op=OP_ECHO_FAST, args=(f"s{i}",)))
                    sync_results.append(r.result)
            except Exception as e:
                errors.append(e)

        async def do_async():
            for i in range(n):
                r = await session.call_async(Request(op=OP_ECHO_FAST, args=(f"a{i}",)))
                async_results.append(r.result)

        t = threading.Thread(target=do_sync)
        t.start()
        asyncio.run(do_async())
        t.join(timeout=10.0)
        assert not t.is_alive() and not errors, errors
        assert sync_results == [f"s{i}" for i in range(n)]
        assert async_results == [f"a{i}" for i in range(n)]
        assert session._pending == {}
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_shutdown_with_inflight_calls_fails_them_cleanly():
    """Shutdown must drain pending callers via the reader's finally —
    in-flight awaiters raise ``SubprocessTransportError``, not hang.
    """
    session = _start_session()
    tasks = [
        asyncio.create_task(session.call_async(Request(op=OP_ASYNC_SLEEP, args=(2.0,))))
        for _ in range(5)
    ]
    deadline = time.monotonic() + 2.0
    while len(session._pending) < 5 and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert len(session._pending) == 5

    # Run shutdown without blocking the event loop.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, session.shutdown)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, SubprocessTransportError) for r in results), results
    assert session._pending == {}


@pytest.mark.asyncio
async def test_late_response_after_caller_timeout_is_dropped():
    """If the worker responds AFTER the caller timed out, the reader
    silently drops the response (the request_id is no longer in
    ``_pending``). Verifies no log spam / no second TimeoutError fires.
    """
    session = _start_session()
    try:
        # Submit a short async sleep with an even shorter timeout —
        # the response WILL arrive shortly after the timeout.
        with pytest.raises(TimeoutError):
            await session.call_async(Request(op=OP_ASYNC_SLEEP, args=(0.3,)), timeout=0.1)
        # Let the late response land.
        await asyncio.sleep(0.4)
        assert session._pending == {}
        # Session still usable.
        resp = await session.call_async(Request(op=OP_ECHO_FAST, args=("after",)))
        assert resp.result == "after"
    finally:
        session.shutdown()
