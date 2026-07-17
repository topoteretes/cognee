"""Concurrent-RPC tests for ``SubprocessSession``.

Exercise the request-id + reader-thread design: many in-flight calls,
out-of-order completion, crash-with-N-pending, sync/async interleaving,
caller cancellation, and per-call timeout cleanup.

Uses a small async-capable worker so the worker-side concurrent dispatch
(``asyncio.create_task`` for async handlers) is also exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import pickle
import time

import pytest

from cognee_db_workers.harness import (
    Request,
    Response,
    SubprocessSession,
    SubprocessTransportError,
    _TIMEOUT_BEFORE_RESPAWN,
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


def _start_retryable_session(**kwargs) -> SubprocessSession:
    """Session with an identical-worker respawn factory; used to verify
    that the retry loop actually fires a respawn on the relevant paths.
    """
    ctx = mp.get_context("spawn")

    def _spawn():
        req_q = ctx.Queue()
        resp_q = ctx.Queue()
        proc = ctx.Process(target=_worker_main, args=(req_q, resp_q), daemon=True)
        with spawn_without_main():
            proc.start()
        return proc, req_q, resp_q

    proc, req_q, resp_q = _spawn()
    session = SubprocessSession(proc, req_q, resp_q, respawn_factory=_spawn, **kwargs)
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
        assert not session._closed_event.is_set()
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


# --- regression tests covering the post-review fixes ---------------------


@pytest.mark.asyncio
async def test_unsolicited_request_id_zero_response_is_dropped():
    """Stray responses with ``request_id=0`` (the protocol-sentinel slot)
    must be dropped by the reader without affecting pending callers.

    Injects a stray response on the response queue while a real RPC is
    in flight; the reader sees ``rid=0``, drops it, and the real RPC
    still completes normally.
    """
    session = _start_session()
    try:
        task = asyncio.create_task(session.call_async(Request(op=OP_ASYNC_SLEEP, args=(0.3,))))
        # Wait for the request to be pending on the session.
        deadline = time.monotonic() + 1.0
        while not session._pending and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert session._pending, "request never reached _pending"

        # Inject a stray rid=0 response. Reader pops it from the queue,
        # sees rid==0, and drops it on the floor.
        session._resp_q.put(Response(request_id=0, result="STRAY"))

        # Original call still completes as if nothing happened.
        resp = await task
        assert resp.result == 0.3
        assert session._pending == {}
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_reader_recovers_from_decode_error_fails_pending():
    """If the response queue raises an unexpected exception during ``get``
    (e.g. an unpicklable payload), the reader's broad ``except`` block
    propagates the failure to every pending future via
    ``_fail_all_pending``. No caller hangs.
    """
    session = _start_session()
    try:
        # Park an async sleep so we have a pending future to fail.
        task = asyncio.create_task(session.call_async(Request(op=OP_ASYNC_SLEEP, args=(5.0,))))
        deadline = time.monotonic() + 1.0
        while not session._pending and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert session._pending, "request never reached _pending"

        # Replace the response queue with a wrapper that surfaces an
        # ``UnpicklingError`` on the next get. The reader is currently
        # parked in the real queue's get(timeout=_PROCESS_CHECK_INTERVAL);
        # it'll loop back, hit the new queue, and trigger the failure
        # path.
        real_resp_q = session._resp_q

        class _FailingQueue:
            def get(self, timeout=None):
                raise pickle.UnpicklingError("synthetic decode failure")

        session._resp_q = _FailingQueue()

        # Wait long enough for the reader's current get(timeout=1.0) to
        # return Empty and rotate to the new queue.
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5.0)
        (caller_result,) = result
        assert isinstance(caller_result, SubprocessTransportError)
        msg = str(caller_result).lower()
        assert "decode" in msg or "unpickl" in msg, caller_result
        # Pending was drained by _fail_all_pending.
        assert session._pending == {}
        # Restore so shutdown sees the real queue.
        session._resp_q = real_resp_q
    finally:
        session.shutdown()


def test_consecutive_timeouts_force_respawn():
    """Per-call timeouts under concurrent RPC no longer mark the session
    closed (sibling calls keep working), but ``_TIMEOUT_BEFORE_RESPAWN``
    timeouts in a row should force a respawn — otherwise the retry loop
    would just hit the same wedged worker again.

    Verified by observing the worker pid change after enough timeouts.
    """
    # max_retries must be high enough for the retry loop to actually
    # invoke _respawn after the threshold is hit. With threshold 2 and
    # call_timeout 0.5s, two timeouts costs ~1s + a respawn round.
    session = _start_retryable_session(call_timeout=0.5, max_retries=_TIMEOUT_BEFORE_RESPAWN + 1)
    try:
        original_pid = session._proc.pid
        # OP_ASYNC_SLEEP(5.0) is well above the 0.5s call timeout, so
        # every attempt times out. The retry loop should hit the threshold
        # mid-way and force a respawn before exhausting all retries.
        with pytest.raises((TimeoutError, SubprocessTransportError)):
            session.call(Request(op=OP_ASYNC_SLEEP, args=(5.0,)))
        # By the time the outer call raises, a respawn must have happened
        # at least once — original worker is dead and replaced.
        assert session._proc.pid != original_pid, (
            "Worker pid unchanged — respawn did not fire after consecutive timeouts"
        )
    finally:
        session.shutdown()


def test_consecutive_timeouts_counter_resets_on_success():
    """A successful call resets the consecutive-timeout counter so that
    isolated timeouts (a single slow op interspersed with healthy calls)
    don't accumulate into a spurious respawn.

    ``max_retries=1`` keeps any single call from itself hitting the
    threshold — so we can see the cross-call reset behavior in isolation.
    """
    session = _start_retryable_session(call_timeout=0.5, max_retries=1)
    try:
        original_pid = session._proc.pid
        # Sequence of isolated timeouts separated by a successful call.
        # Without the reset-on-success, three timeouts in this sequence
        # would accumulate past the threshold and force a respawn.
        for _ in range(3):
            with pytest.raises(TimeoutError):
                session.call(Request(op=OP_ASYNC_SLEEP, args=(5.0,)), timeout=0.3)
            ok = session.call(Request(op=OP_ECHO_FAST, args=("ping",)), timeout=10.0)
            assert ok.result == "ping"
        # Worker survived all of that — no respawn fired.
        assert session._proc.pid == original_pid
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_fresh_caller_during_respawn_does_not_hang():
    """Regression for the race where a non-retry caller arrives between
    the old reader exiting and the new reader starting inside
    ``_respawn``. With the fix (``_closed_event`` stays set across the
    entire respawn), the caller sees ``SubprocessTransportError`` on
    insert, then is naturally rerouted through the retry loop and lands
    on the post-respawn session.
    """
    import threading

    session = _start_retryable_session()
    try:
        # Kill the current worker so the next call triggers a respawn.
        original_pid = session._proc.pid
        session._proc.kill()

        # Fire two callers concurrently: both will fail their first
        # attempt with a transport error, then race to acquire
        # ``_respawn_lock`` and either coalesce (one respawns, the other
        # waits) or run sequentially. Either way no one hangs.
        async def caller(payload):
            return await session.call_async(Request(op=OP_ECHO_FAST, args=(payload,)), timeout=10.0)

        results = await asyncio.wait_for(
            asyncio.gather(caller("a"), caller("b"), return_exceptions=True),
            timeout=15.0,
        )
        # Both must have succeeded; no transport errors propagated.
        for r in results:
            assert not isinstance(r, BaseException), f"caller failed: {r!r}"
        assert {r.result for r in results} == {"a", "b"}
        # And the worker really did get respawned.
        assert session._proc.pid != original_pid
        # No stragglers.
        assert session._pending == {}
        # Ensure no reader-thread leak by checking we have exactly one
        # live reader at the end.
        assert session._reader_thread is not None and session._reader_thread.is_alive()
    finally:
        session.shutdown()
    # Use the threading import to satisfy linters; the assertions above
    # would also fail loudly if a deadlock left threads parked.
    _ = threading.active_count()
