"""Regression test for issue #3708 (alternate async-resolution fix): re-creating
a subprocess graph engine for a path whose previous worker is still shutting
down must NOT fail with "Could not set lock on file (Lock is held by PID X)".

The previous worker held the on-disk file lock for its whole lifetime; a new
worker opened the same path before the old one had exited. The fix routes engine
creation through an async cache-acquisition path (`acreate_graph_engine` ->
`_create_graph_engine.acall` -> `aget_or_create`) that awaits any in-flight close
of the same cache key before constructing, waits for the worker process to
actually exit, and retries the open on transient lock contention.

Drives the REAL graph-engine cache in subprocess mode — no LLM/full config needed.
"""

from __future__ import annotations

import asyncio
import gc
import os

import pytest

pytest.importorskip("ladybug")

from cognee.infrastructure.databases.graph.get_graph_engine import (
    acreate_graph_engine,
    evict_graph_engine,
)


def _config(tmp_path) -> dict:
    return dict(
        graph_database_provider="ladybug",
        graph_file_path=os.path.join(str(tmp_path), "graphdir"),
        graph_database_subprocess_enabled=True,
        kuzu_buffer_pool_size=1 << 28,
        kuzu_max_db_size=1 << 30,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@pytest.mark.asyncio
async def test_async_recreate_after_evict_no_lock_error(tmp_path):
    """Repeated evict -> re-acquire cycles for the same path via the async
    acquisition path must never hit the lock error."""
    cfg = _config(tmp_path)
    for _ in range(5):
        engine = await acreate_graph_engine(**cfg)
        await engine.query("MATCH (n) RETURN 1 LIMIT 1")
        del engine
        evict_graph_engine(**cfg)
        gc.collect()
    engine = await acreate_graph_engine(**cfg)
    await engine.close()


@pytest.mark.asyncio
async def test_concurrent_recreate_after_evict_single_worker(tmp_path):
    """Many coroutines concurrently re-acquiring the same path while the prior
    worker is mid-close must all wait on that one close and converge on a single
    new worker — no lock error, no thundering herd of workers racing the lock.

    Exercises the real subprocess worker (not stub closeables), the gap both the
    sync- and async-approach test suites otherwise leave uncovered.
    """
    cfg = _config(tmp_path)
    engine = await acreate_graph_engine(**cfg)
    await engine.query("MATCH (n) RETURN 1 LIMIT 1")
    old_pid = engine._session.pid
    del engine
    evict_graph_engine(**cfg)  # close starts; worker still shutting down
    gc.collect()

    engines = await asyncio.gather(*[acreate_graph_engine(**cfg) for _ in range(8)])

    pids = {e._session.pid for e in engines}
    assert len(pids) == 1, f"expected a single live worker, got {pids}"
    assert old_pid not in pids, "must be a fresh worker, not the closed one"
    # The single resolved engine is functional (lock was acquired cleanly).
    await engines[0].query("MATCH (n) RETURN 1 LIMIT 1")
    await engines[0].close()


@pytest.mark.asyncio
async def test_held_engine_reference_survives_eviction(tmp_path):
    """A held engine reference must keep working after the engine is evicted
    from the cache — which is what dataset-context teardown
    (``_teardown_subprocess_engines``) does on context exit. The lease defers
    the close until the reference is dropped rather than force-closing it, so
    reuse returns a stale-but-live adapter instead of raising "adapter is
    closed". Regression for the CI failures where reusing a ``get_*_engine()``
    result after prune/delete/context-exit crashed.
    """
    cfg = _config(tmp_path)
    engine = await acreate_graph_engine(**cfg)
    await engine.query("MATCH (n) RETURN 1 LIMIT 1")

    # Simulate what _teardown_subprocess_engines does on context exit: evict
    # (detach), NOT force-close. The held `engine` reference keeps it alive.
    evict_graph_engine(**cfg)
    gc.collect()

    # Reuse must still work (stale-but-live), NOT raise "adapter is closed".
    await engine.query("MATCH (n) RETURN 1 LIMIT 1")

    # Dropping the reference releases the lease → close fires; a fresh resolve
    # for the same path then awaits that close and opens cleanly.
    del engine
    gc.collect()
    fresh = await acreate_graph_engine(**cfg)
    await fresh.query("MATCH (n) RETURN 1 LIMIT 1")
    await fresh.close()


@pytest.mark.asyncio
async def test_close_waits_for_worker_process_exit(tmp_path):
    """``close()`` must not return until the worker process has actually exited
    (and thus released its on-disk file lock)."""
    cfg = _config(tmp_path)
    engine = await acreate_graph_engine(**cfg)
    await engine.query("MATCH (n) RETURN 1 LIMIT 1")
    pid = engine._session.pid
    assert _pid_alive(pid)

    await engine.close()

    assert not _pid_alive(pid), f"worker pid {pid} still alive after close()"
