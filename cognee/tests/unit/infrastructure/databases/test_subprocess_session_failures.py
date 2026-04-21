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
import os
import signal
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


def _echo(registry, req):
    return req.args[0]


def _sleep(registry, req):
    # Sleep forever — simulates a hung native call.
    time.sleep(60.0)
    return None


class _NotPicklable:
    def __reduce__(self):  # pragma: no cover - deliberately broken
        raise TypeError("nope")


def _return_unpicklable(registry, req):
    return _NotPicklable()


DISPATCH = {
    OP_ECHO: _echo,
    OP_SLEEP: _sleep,
    OP_RETURN_UNPICKLABLE: _return_unpicklable,
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
        proc, req_q, resp_q,
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


def test_worker_killed_mid_request_flips_closed():
    # No respawn factory → no retry. Verify the session surfaces the
    # transport error and flips closed.
    session = _start_session()
    try:
        os.kill(session.pid, signal.SIGKILL)
        deadline = time.time() + 5
        while session._proc.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        assert not session._proc.is_alive(), "worker did not die after SIGKILL"

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
    proc = ctx.Process(
        target=_failing_init_worker, args=(req_q, resp_q), daemon=True
    )
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
    rows = await adapter.query("RETURN 1 AS x")
    assert rows == [(1,)]

    await adapter.close()

    with pytest.raises(RuntimeError, match="closed"):
        await adapter.query("RETURN 1")


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

        os.kill(session.pid, signal.SIGKILL)
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
            proc = ctx.Process(
                target=_stillborn_worker, args=(req_q, resp_q), daemon=True
            )
            with spawn_without_main():
                proc.start()
            return proc, req_q, resp_q

        session._respawn_factory = _stillborn_spawn

        os.kill(session.pid, signal.SIGKILL)
        while session._proc.is_alive():
            time.sleep(0.05)

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

        session.add_replay_step(
            ReplayStep(make_request=_make_replay_req, apply_new_handle=None)
        )

        os.kill(session.pid, signal.SIGKILL)
        while session._proc.is_alive():
            time.sleep(0.05)

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
        await adapter.query(
            "CREATE NODE TABLE IF NOT EXISTS N(id STRING PRIMARY KEY, name STRING)"
        )
        await adapter.query(
            "CREATE (:N {id: $id, name: $name})",
            {"id": "a", "name": "Alice"},
        )
        rows = await adapter.query("MATCH (n:N) RETURN n.id, n.name")
        assert rows == [("a", "Alice")]

        # Now SIGKILL the worker.
        session = adapter._session
        old_pid = session.pid
        os.kill(session.pid, signal.SIGKILL)
        while session._proc.is_alive():
            time.sleep(0.05)

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
        conn = RemoteLanceDBConnection(
            session, url=str(tmp_path / "lance"), api_key=None
        )
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
        with pytest.raises(AssertionError):
            _ = t.handle_id
    finally:
        session.shutdown()
