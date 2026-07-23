"""Regression test for issue #3708: re-creating a subprocess graph engine for a
path whose previous worker is still shutting down must NOT fail with
"Could not set lock on file (Lock is held by PID ...)".

Root cause: the previous worker held the on-disk file lock for its whole
lifetime; a new worker opened the same path before the old one had exited. The
fix coordinates create-vs-close through the closing registry (async acquisition
waits for the in-flight close), closes subprocess adapters off the event loop so
the lock is released even when a sync re-resolution blocks the loop, waits for
the worker process to actually exit, and retries the open on transient lock
contention.

Drives the REAL graph-engine cache (`create_graph_engine` / `acreate_graph_engine`
/ `evict_graph_engine`) in subprocess mode — no LLM or full cognee config needed.
"""

from __future__ import annotations

import gc
import os

import pytest

pytest.importorskip("ladybug")

from cognee.infrastructure.databases.graph.get_graph_engine import (
    acreate_graph_engine,
    create_graph_engine,
    evict_graph_engine,
)


def _config(tmp_path) -> dict:
    return dict(
        graph_database_provider="ladybug",
        graph_file_path=os.path.join(str(tmp_path), "graphdir"),
        graph_database_subprocess_enabled=True,
        # Small pools keep the worker cheap to spawn in tests.
        kuzu_buffer_pool_size=1 << 28,
        kuzu_max_db_size=1 << 30,
    )


async def _evict_after_use(cfg, *, use_async):
    """Resolve an engine, run a query, then drop the reference and evict — so
    the previous worker's close starts with no caller holding it (mirrors the
    engine handle dropping a stale pin)."""
    engine = await acreate_graph_engine(**cfg) if use_async else create_graph_engine(**cfg)
    await engine.query("MATCH (n) RETURN 1 LIMIT 1")
    del engine
    evict_graph_engine(**cfg)
    gc.collect()


@pytest.mark.asyncio
async def test_async_recreate_after_evict_no_lock_error(tmp_path):
    """The async acquisition path must serialize cleanly across many
    evict→recreate cycles for the same path."""
    cfg = _config(tmp_path)
    for _ in range(5):
        await _evict_after_use(cfg, use_async=True)
    # Cleanly tear down the last live engine.
    engine = await acreate_graph_engine(**cfg)
    await engine.close()


@pytest.mark.asyncio
async def test_sync_recreate_after_evict_no_lock_error(tmp_path):
    """The sync re-resolution path blocks the loop while opening the new worker;
    the off-loop close + worker open-retry must still let it succeed."""
    cfg = _config(tmp_path)
    for _ in range(5):
        await _evict_after_use(cfg, use_async=False)
    engine = create_graph_engine(**cfg)
    await engine.close()


@pytest.mark.asyncio
async def test_handle_reresolves_after_eviction(tmp_path):
    """A held engine handle must transparently re-resolve after its cached
    engine is evicted (as dataset-context teardown does), instead of raising
    "adapter is closed". Regression for the CI failures where reusing a
    get_graph_engine() result across prune/delete/context-exit crashed."""
    from cognee.infrastructure.databases.graph.get_graph_engine import _GraphEngineHandle

    cfg = _config(tmp_path)
    handle = _GraphEngineHandle(cfg)
    await handle.query("MATCH (n) RETURN 1 LIMIT 1")

    # Simulate teardown: evict the cached engine. The handle pins the proxy, so
    # its next access detects the stale pin, drops it (deferred close releases
    # the lock off-loop), and re-resolves a fresh engine for the same path.
    evict_graph_engine(**cfg)
    gc.collect()

    await handle.query("MATCH (n) RETURN 1 LIMIT 1")  # must NOT raise "is closed"
    await handle.close()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@pytest.mark.asyncio
async def test_close_waits_for_worker_process_exit(tmp_path):
    """``close()`` must not return until the worker process has actually exited
    (and thus released its on-disk file lock) — the guarantee that lets a new
    worker open the same path immediately afterwards."""
    cfg = _config(tmp_path)
    engine = create_graph_engine(**cfg)
    await engine.query("MATCH (n) RETURN 1 LIMIT 1")
    pid = engine._session.pid
    assert _pid_alive(pid)

    await engine.close()

    assert not _pid_alive(pid), f"worker pid {pid} still alive after close()"
