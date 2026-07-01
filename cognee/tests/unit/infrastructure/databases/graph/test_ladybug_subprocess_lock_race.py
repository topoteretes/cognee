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
