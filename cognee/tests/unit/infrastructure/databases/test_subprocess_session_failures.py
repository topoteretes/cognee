"""Failure-path tests for the subprocess harness.

Covers the scenarios that used to silently hang or leak:
- Worker killed mid-request
- Init timeout
- Per-call timeout on a hung worker
- Non-picklable return value from the worker
- Table-handle release on ``RemoteLanceDBTable``
- Closed-adapter reuse is a hard error in both Kuzu and LanceDB adapters
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from cognee_db_workers.harness import (
    ReplayStep,
    Request,
    Response,
    SubprocessSession,
    SubprocessTransportError,
    run_worker_loop,
    spawn_without_main,
)


# --- helper workers -------------------------------------------------------


OP_ECHO = 1
OP_SLEEP = 2
OP_RETURN_UNPICKLABLE = 3
OP_SLEEP_PARAM = 4
OP_RAISE_PICKLABLE = 5


def _echo(registry, req):
    return req.args[0]


def _sleep(registry, req):
    # Sleep forever — simulates a hung native call.
    time.sleep(60.0)
    return None


def _sleep_param(registry, req):
    # Bounded sleep used by race-window tests. Caller sets the duration.
    time.sleep(req.args[0])
    return None


class _NotPicklable:
    def __reduce__(self):  # pragma: no cover - deliberately broken
        raise TypeError("nope")


def _return_unpicklable(registry, req):
    return _NotPicklable()


def _raise_picklable(registry, req):
    # Raise an ordinary picklable exception with a deep call stack so the
    # remote traceback is non-trivial. Helper functions at module scope
    # so the inner frames show up in the traceback string.
    _layer_a()


def _layer_a():
    _layer_b()


def _layer_b():
    raise ValueError("worker-side boom")


DISPATCH = {
    OP_ECHO: _echo,
    OP_SLEEP: _sleep,
    OP_SLEEP_PARAM: _sleep_param,
    OP_RETURN_UNPICKLABLE: _return_unpicklable,
    OP_RAISE_PICKLABLE: _raise_picklable,
}


def _worker_main(req_q, resp_q):
    run_worker_loop(DISPATCH, req_q, resp_q)


def _init_that_raises(registry):
    raise RuntimeError("boom during init")


def _failing_init_worker(req_q, resp_q):
    run_worker_loop(DISPATCH, req_q, resp_q, init=_init_that_raises)


def _never_ready_worker(req_q, resp_q):
    # Never emits the ready sentinel; caller is expected to time out.
    time.sleep(60)


def _stillborn_worker(req_q, resp_q):
    # Exits immediately without emitting the ready sentinel. Used to simulate
    # a respawn factory that keeps handing back dead workers.
    return


def _wait_for_worker_exit(session, timeout: float = 5.0) -> None:
    """Poll until the worker process is no longer alive, bounded by ``timeout``.

    Several tests SIGKILL the worker and then need to observe its death
    before issuing the next RPC. An unbounded wait hangs the suite if the
    kill signal is delivered out-of-order or the process never transitions
    (e.g. because it was reaped by another test's cleanup path).
    """
    deadline = time.time() + timeout
    while session._proc.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    assert not session._proc.is_alive(), f"worker did not exit within {timeout}s"


def _start_session(target=_worker_main, **kwargs) -> SubprocessSession:
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    resp_q = ctx.Queue()
    proc = ctx.Process(target=target, args=(req_q, resp_q), daemon=True)
    with spawn_without_main():
        proc.start()
    session = SubprocessSession(proc, req_q, resp_q, **kwargs)
    session.wait_for_ready()
    return session


def _start_retryable_session(
    target=_worker_main,
    max_retries: int = 2,
    init_timeout: float = 10.0,
) -> SubprocessSession:
    """Session whose ``respawn_factory`` spawns an identical fresh worker,
    used to exercise the retry + replay machinery.
    """
    ctx = mp.get_context("spawn")

    def _spawn():
        req_q = ctx.Queue()
        resp_q = ctx.Queue()
        proc = ctx.Process(target=target, args=(req_q, resp_q), daemon=True)
        with spawn_without_main():
            proc.start()
        return proc, req_q, resp_q

    proc, req_q, resp_q = _spawn()
    session = SubprocessSession(
        proc,
        req_q,
        resp_q,
        respawn_factory=_spawn,
        max_retries=max_retries,
        init_timeout=init_timeout,
    )
    session.wait_for_ready()
    return session


# --- tests ---------------------------------------------------------------


def test_call_returns_echo():
    session = _start_session()
    try:
        resp = session.call(Request(op=OP_ECHO, args=("hello",)))
        assert resp.result == "hello"
    finally:
        session.shutdown()


# Note on monotonic-clock testing: ``_resolve_deadline`` and
# ``_wait_response`` switched from ``time.time()`` to ``time.monotonic()``
# (Copilot review comment 3167236652) so deadline math can't be skewed by
# wall-clock jumps. We deliberately don't add a unit test for this: a
# meaningful test would have to monkeypatch ``time.monotonic`` (or both
# clocks together) and inject a clock jump mid-call, which is invasive
# and brittle. The change is a clean swap to a strictly-monotonic source,
# verifiable by code review; the existing timeout/deadline tests
# (``test_call_timeout_on_hung_worker``, ``test_per_call_timeout_override``)
# continue to exercise the deadline path with the new clock domain.


def test_picklable_worker_exception_carries_remote_traceback():
    """When the worker raises a picklable exception, the local caller
    should see the same exception type AND have access to the remote
    traceback via ``__notes__`` (PEP 678). Without preservation, the
    worker-side stack frames are lost and debugging gets harder.
    """
    session = _start_session()
    try:
        with pytest.raises(ValueError, match="worker-side boom") as excinfo:
            session.call(Request(op=OP_RAISE_PICKLABLE))
        notes = getattr(excinfo.value, "__notes__", []) or []
        joined = "\n".join(notes)
        # Note: ``add_note`` is Python 3.11+. On 3.10 the test still
        # asserts the exception type/message but skips the notes check.
        if hasattr(excinfo.value, "add_note"):
            assert "Remote subprocess traceback:" in joined, (
                "remote traceback annotation missing from exception notes"
            )
            assert "_layer_b" in joined, "remote traceback should include worker-side stack frames"
    finally:
        session.shutdown()


def test_worker_killed_mid_request_flips_closed():
    # No respawn factory → no retry. Verify the session surfaces the
    # transport error and flips closed.
    session = _start_session()
    try:
        session._proc.kill()
        deadline = time.time() + 5
        while session._proc.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        assert not session._proc.is_alive(), "worker did not die after kill()"

        with pytest.raises(SubprocessTransportError, match="Subprocess exited unexpectedly"):
            session.call(Request(op=OP_ECHO, args=("hi",)))
        # Session should now be flipped to closed; next call gets the faster,
        # explicit "session is closed" message.
        assert session._closed is True
        with pytest.raises(SubprocessTransportError, match="session is closed"):
            session.call(Request(op=OP_ECHO, args=("hi",)))
    finally:
        session.shutdown()


def test_init_timeout_raises():
    """A worker that never puts the ready sentinel times out cleanly."""
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    resp_q = ctx.Queue()
    proc = ctx.Process(target=_never_ready_worker, args=(req_q, resp_q), daemon=True)
    with spawn_without_main():
        proc.start()
    session = SubprocessSession(proc, req_q, resp_q, init_timeout=1.0)
    with pytest.raises(RuntimeError, match="init timed out"):
        session.wait_for_ready()
    assert session._closed is True


def test_init_failure_propagates():
    """If the worker's init callback raises, ``wait_for_ready`` should say so."""
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    resp_q = ctx.Queue()
    proc = ctx.Process(target=_failing_init_worker, args=(req_q, resp_q), daemon=True)
    with spawn_without_main():
        proc.start()
    session = SubprocessSession(proc, req_q, resp_q, init_timeout=5.0)
    with pytest.raises(RuntimeError, match="init failed"):
        session.wait_for_ready()
    assert session._closed is True


def test_call_timeout_on_hung_worker():
    """A request that doesn't return within the deadline raises TimeoutError
    and marks the session closed so callers don't wait forever.
    """
    session = _start_session(call_timeout=1.0)
    try:
        with pytest.raises(TimeoutError):
            session.call(Request(op=OP_SLEEP, args=()))
        assert session._closed is True
    finally:
        session.shutdown()


def test_per_call_timeout_override():
    session = _start_session(call_timeout=60.0)
    try:
        # Override the session default for this one call
        with pytest.raises(TimeoutError):
            session.call(Request(op=OP_SLEEP, args=()), timeout=0.5)
    finally:
        session.shutdown()


def test_unpicklable_return_surfaces_error():
    session = _start_session()
    try:
        # The worker's pickle of the Response will fail when putting on the
        # queue. mp.Queue raises at put time. We just want to ensure it
        # doesn't hang the session.
        with pytest.raises(Exception):
            session.call(Request(op=OP_RETURN_UNPICKLABLE, args=()), timeout=5.0)
    finally:
        session.shutdown()


# --- adapter integration -------------------------------------------------


@pytest.mark.asyncio
async def test_kuzu_adapter_rejects_use_after_close(tmp_path):
    from cognee.infrastructure.databases.graph.get_graph_engine import (
        create_graph_engine,
    )

    adapter = create_graph_engine(
        graph_database_provider="kuzu",
        graph_file_path=str(tmp_path / "kz"),
        graph_database_subprocess_enabled=True,
    )
    try:
        rows = await adapter.query("RETURN 1 AS x")
        assert rows == [(1,)]

        await adapter.close()

        with pytest.raises(RuntimeError, match="closed"):
            await adapter.query("RETURN 1")
    finally:
        # Idempotent close — guards against early assertion failure
        # leaking the subprocess-backed worker into other tests.
        try:
            await adapter.close()
        except Exception:
            pass


# --- retry / replay -------------------------------------------------------


def test_retry_on_worker_sigkill_succeeds():
    """Kill the worker mid-request; the session respawns and the retry
    completes on the new worker.
    """
    session = _start_retryable_session(max_retries=2)
    try:
        # Prove baseline works
        assert session.call(Request(op=OP_ECHO, args=("a",))).result == "a"
        original_pid = session.pid

        session._proc.kill()
        # Give it a moment so the next call trips is_alive checks, not the
        # faster "already dead on entry" path.
        deadline = time.time() + 3
        while session._proc.is_alive() and time.time() < deadline:
            time.sleep(0.05)

        resp = session.call(Request(op=OP_ECHO, args=("after-respawn",)))
        assert resp.result == "after-respawn"
        assert session.pid != original_pid, "session should have respawned"
    finally:
        session.shutdown()


def test_retry_gives_up_after_max_retries():
    """If every respawn attempt also fails (here: the factory hands back a
    process that exits before sending the ready sentinel), the session
    eventually surfaces the transport error rather than looping forever.
    """
    # Tight init_timeout so the test doesn't sit on the default 60s.
    session = _start_retryable_session(max_retries=1, init_timeout=2.0)
    try:
        ctx = mp.get_context("spawn")

        def _stillborn_spawn():
            req_q = ctx.Queue()
            resp_q = ctx.Queue()
            proc = ctx.Process(target=_stillborn_worker, args=(req_q, resp_q), daemon=True)
            with spawn_without_main():
                proc.start()
            return proc, req_q, resp_q

        session._respawn_factory = _stillborn_spawn

        session._proc.kill()
        _wait_for_worker_exit(session)

        with pytest.raises(SubprocessTransportError):
            session.call(Request(op=OP_ECHO, args=("never",)))
    finally:
        session.shutdown()


def test_retry_does_not_fire_on_application_error():
    """Errors raised inside the worker's handler should NOT trigger a respawn
    — the retry would deterministically fail the same way and waste a full
    worker-start cycle.
    """
    session = _start_retryable_session(max_retries=3)
    try:
        original_pid = session.pid
        # OP 9999 is not registered → the worker returns a Response with
        # error="Unknown op 9999" (application error, not transport).
        with pytest.raises(RuntimeError, match="Unknown op"):
            session.call(Request(op=9999))
        assert session.pid == original_pid, "should NOT have respawned"
        # Session still usable after an application error.
        assert session.call(Request(op=OP_ECHO, args=("still-ok",))).result == "still-ok"
    finally:
        session.shutdown()


def test_replay_steps_fire_on_respawn():
    """Registered replay steps run (in registration order) against the
    freshly spawned worker before the original failing RPC is retried.
    Handle-remap bookkeeping is exercised by the full Kuzu integration
    test below; here we just verify the replay hook runs.
    """
    session = _start_retryable_session(max_retries=2)
    try:
        replay_invocations = []

        def _make_replay_req():
            replay_invocations.append("fired")
            return Request(op=OP_ECHO, args=("replay-run",))

        session.add_replay_step(ReplayStep(make_request=_make_replay_req, apply_new_handle=None))

        session._proc.kill()
        _wait_for_worker_exit(session)

        # First call after crash triggers: respawn → replay (one step) →
        # retry the original request on the new worker.
        resp = session.call(Request(op=OP_ECHO, args=("after",)))
        assert resp.result == "after"
        assert replay_invocations == ["fired"], (
            "Replay step must run exactly once during the respawn"
        )
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_kuzu_adapter_survives_worker_sigkill(tmp_path):
    """End-to-end: kill the Kuzu subprocess mid-test; the adapter keeps
    working because the session respawns + replays the database/connection
    setup + retries the failing query.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import (
        create_graph_engine,
    )

    adapter = create_graph_engine(
        graph_database_provider="kuzu",
        graph_file_path=str(tmp_path / "kz"),
        graph_database_subprocess_enabled=True,
    )
    try:
        # Sanity: create a table + insert.
        await adapter.query("CREATE NODE TABLE IF NOT EXISTS N(id STRING PRIMARY KEY, name STRING)")
        await adapter.query(
            "CREATE (:N {id: $id, name: $name})",
            {"id": "a", "name": "Alice"},
        )
        rows = await adapter.query("MATCH (n:N) RETURN n.id, n.name")
        assert rows == [("a", "Alice")]

        # Now SIGKILL the worker.
        session = adapter._session
        old_pid = session.pid
        session._proc.kill()
        _wait_for_worker_exit(session)

        # Next query should respawn + replay + retry transparently. The Kuzu
        # DB on disk still has our row, so we expect it back.
        rows = await adapter.query("MATCH (n:N) RETURN n.id, n.name")
        assert rows == [("a", "Alice")]
        assert session.pid != old_pid
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_lancedb_table_handle_release(tmp_path):
    """Opening a table allocates a worker-side handle; dropping the proxy
    must release it so the worker's handle dict doesn't grow unboundedly.
    """
    import pyarrow as pa

    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        LanceDBSubprocessSession,
        RemoteLanceDBConnection,
    )

    session = LanceDBSubprocessSession.start()
    try:
        conn = RemoteLanceDBConnection(session, url=str(tmp_path / "lance"), api_key=None)
        await conn.connect()
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), 4)),
            ]
        )
        t = await conn.create_table("t", schema=schema, exist_ok=True)
        handle_id = t.handle_id
        assert handle_id is not None

        await t.release()
        # Releasing again is a no-op
        await t.release()

        # handle_id is now cleared; attempting to use the table fails loudly
        # (RuntimeError, not AssertionError — asserts get stripped under -O).
        with pytest.raises(RuntimeError, match="lancedb table handle released"):
            _ = t.handle_id
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_lancedb_table_apply_new_handle_after_release_does_not_resurrect(tmp_path):
    """Race-window guard: if ``_respawn`` snapshotted ``_replay_steps`` before
    ``release()`` deregistered ours, the snapshot will replay our OPEN_TABLE
    step and fire ``_apply_new_handle`` on a proxy whose ``_handle_id`` is
    already ``None``. The callback must not write the new id back —
    otherwise the released table is silently resurrected.
    """
    import pyarrow as pa

    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        LanceDBSubprocessSession,
        RemoteLanceDBConnection,
    )

    session = LanceDBSubprocessSession.start()
    try:
        conn = RemoteLanceDBConnection(session, url=str(tmp_path / "lance"), api_key=None)
        await conn.connect()
        schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), 4))])
        t = await conn.create_table("t", schema=schema, exist_ok=True)
        await t.release()
        assert t._handle_id is None

        # Simulate the racing replay: callback fires with a fresh handle id.
        result = t._apply_new_handle(99999)
        assert result is None, "released proxy must not register a remap entry"
        assert t._handle_id is None, "released proxy must not be resurrected"
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_lancedb_connection_concurrent_connect_registers_one_replay_step(tmp_path):
    """Concurrent ``connect()`` callers must not each append their own
    OP_CONNECT replay step. Without the lock, every coroutine that
    observes ``_connected == False`` before the first one returns will
    register a duplicate, and every respawn then reconnects N times.
    """
    import asyncio as _asyncio

    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        LanceDBSubprocessSession,
        RemoteLanceDBConnection,
    )
    from cognee_db_workers.lancedb_protocol import OP_CONNECT

    session = LanceDBSubprocessSession.start()
    try:
        conn = RemoteLanceDBConnection(session, url=str(tmp_path / "lance"), api_key=None)

        # 16 concurrent connect() calls. The first to reach the slow path
        # should do the real work; the rest must be no-ops.
        await _asyncio.gather(*(conn.connect() for _ in range(16)))

        connect_steps = [s for s in session._replay_steps if s.make_request().op == OP_CONNECT]
        assert len(connect_steps) == 1, (
            f"expected exactly one OP_CONNECT replay step, got {len(connect_steps)} — "
            f"concurrent connect() calls are not properly serialized"
        )
        assert conn._connected is True
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_lancedb_table_async_with_releases_handle(tmp_path):
    """``async with table:`` must release the worker-side handle on exit.

    Without a proper ``__aexit__``, the handle survives until GC runs and the
    worker's HandleRegistry grows unboundedly across cognify cycles.
    """
    import pyarrow as pa

    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        LanceDBSubprocessSession,
        RemoteLanceDBConnection,
    )

    session = LanceDBSubprocessSession.start()
    try:
        conn = RemoteLanceDBConnection(session, url=str(tmp_path / "lance"), api_key=None)
        await conn.connect()
        schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), 4))])
        async with await conn.create_table("t", schema=schema, exist_ok=True) as t:
            assert t.handle_id is not None
        # Block-exit must have released the handle.
        with pytest.raises(RuntimeError, match="lancedb table handle released"):
            _ = t.handle_id
    finally:
        session.shutdown()


@pytest.mark.asyncio
async def test_lancedb_table_sync_with_releases_handle(tmp_path):
    """Sync ``with table:`` must release the handle on exit, matching
    upstream ``lancedb.AsyncTable.__exit__`` semantics. The proxy is
    constructed via the async public API; the sync exit is what we test."""
    import pyarrow as pa

    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        LanceDBSubprocessSession,
        RemoteLanceDBConnection,
    )

    session = LanceDBSubprocessSession.start()
    try:
        conn = RemoteLanceDBConnection(session, url=str(tmp_path / "lance"), api_key=None)
        await conn.connect()
        schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), 4))])
        t = await conn.create_table("t", schema=schema, exist_ok=True)
        with t:
            assert t.handle_id is not None
        with pytest.raises(RuntimeError, match="lancedb table handle released"):
            _ = t.handle_id
    finally:
        session.shutdown()


def test_concurrent_shutdown_with_inflight_call_does_not_hang():
    """Stress: race ``shutdown()`` against an in-flight ``call()`` on the
    same session.

    Each iteration starts a fresh session, issues a slow
    ``OP_SLEEP_PARAM`` on a background thread, then fires ``shutdown()``
    while the caller is parked inside ``_resp_q.get()``.

    Important caveat about what this test does and does not catch.
    ``multiprocessing.Queue.get()`` holds an internal ``_rlock`` for the
    duration of the receive, so the consumer that calls ``get()`` first
    blocks any other consumer until it returns — they don't actually
    race byte-for-byte for the next item. The narrow remaining race is
    the window between the caller's ``_req_q.put`` and ``_resp_q.get``;
    it's microseconds wide and OS scheduling typically favors the
    already-parked caller. So in practice this test can't reliably
    *reproduce* the protocol-corruption mode the reviewer described
    against an unsynchronized ``shutdown()``.

    What it does verify: shutdown and call interleave safely under
    heavy contention — no thread hangs, every worker is reaped, no
    spurious exceptions surface. That's a useful regression guard
    against future ``shutdown()`` refactors that would, e.g., swallow
    the in-flight caller's get and cause it to wait until its deadline.
    """
    import threading

    n_iterations = 30
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    for _ in range(n_iterations):
        session = _start_session()

        def caller(s=session):
            try:
                # Short sleep — wide enough to be parked in ``get()`` when
                # the shutter arrives, but small enough to keep the test
                # quick. ``timeout=1.0`` so a stolen response surfaces as
                # ``TimeoutError`` within the iteration budget.
                s.call(Request(op=OP_SLEEP_PARAM, args=(0.05,)), timeout=1.0)
            except SubprocessTransportError:
                pass  # acceptable: session was closed mid-flight
            except TimeoutError:
                with errors_lock:
                    errors.append(
                        TimeoutError("call() timed out — response likely stolen by shutter")
                    )
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        def shutter(s=session):
            try:
                s.shutdown(timeout=1.0)
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        ct = threading.Thread(target=caller)
        st = threading.Thread(target=shutter)
        ct.start()
        # Tiny delay to land the caller inside ``_resp_q.get()`` before
        # ``shutdown()`` puts its SHUTDOWN sentinel — but small enough that
        # the caller has not yet received a response.
        time.sleep(0.01)
        st.start()

        ct.join(timeout=5.0)
        st.join(timeout=5.0)
        assert not ct.is_alive() and not st.is_alive(), "race produced a hang"
        assert not session._proc.is_alive()

    assert not errors, f"unexpected exceptions across {n_iterations} iterations: {errors!r}"
