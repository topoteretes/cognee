"""Fail any test in this directory that leaves a subprocess worker running.

Graph and vector adapters run their native engines in spawned worker
processes. A worker that outlives the test that started it is a leak: in a
server it stays alive, about 120 MiB each, until the process exits (#5710).
The suite could not see such a leak before: ``cognee/tests/conftest.py``
reaps leftover workers at session end and said nothing about them.

After each test the fixture compares the live child processes with the
snapshot taken before the test. A new worker is allowed while an engine
cache still holds the adapter that owns it (cached engines legitimately
outlive a test), or when it exits within ``WORKER_EXIT_TIMEOUT_SECONDS`` (a
close the cache started on a pool thread may still be shutting it down).
Anything else fails the test, naming the worker and the session that
spawned it, and is killed so the next test starts clean. No ``gc.collect()``
here on purpose: a worker kept alive only by a reference cycle is the leak
this exists to catch, and collecting the cycle would hide it.

The check watches processes, not the caches' pending-close futures: a
close scheduled on an event loop that a test has already torn down never
resolves, and waiting on it would charge every later test for it.
"""

from __future__ import annotations

import multiprocessing
import time

import pytest

from cognee.infrastructure.databases.utils.closing_lru_cache import _DECORATED_CACHES
from cognee_db_workers.harness import SubprocessSession, _all_sessions

# Bounds a worker's shutdown: the harness waits up to 10s for the SHUTDOWN
# ack before its kill chain, so a close in flight is done well within this.
WORKER_EXIT_TIMEOUT_SECONDS = 20.0


def _live_children() -> dict[int, multiprocessing.Process]:
    return {proc.pid: proc for proc in multiprocessing.active_children() if proc.is_alive()}


def _attribute_objects(value, depth: int):
    """``value``'s attribute values, then theirs, ``depth`` hops deep."""
    frontier = [value]
    for _ in range(depth):
        next_frontier = []
        for obj in frontier:
            for attr in list(vars(obj).values()) if hasattr(obj, "__dict__") else []:
                yield attr
                next_frontier.append(attr)
        frontier = next_frontier


def _sessions_held_by_engine_caches() -> set[SubprocessSession]:
    """Sessions reachable from a value an engine cache still holds.

    LanceDB keeps its session on the adapter; Kuzu keeps it on the database
    and connection proxies the adapter holds, hence two hops.
    """
    held: set[SubprocessSession] = set()
    for cache in list(_DECORATED_CACHES):
        with cache._lock:
            values = [entry.value for entry in cache._cache.values()]
        for value in values:
            for obj in _attribute_objects(value, depth=2):
                if isinstance(obj, SubprocessSession):
                    held.add(obj)
    return held


def _still_alive_after(procs: list[multiprocessing.Process], timeout: float) -> list:
    """Give workers whose close is in flight time to exit; return the rest."""
    deadline = time.monotonic() + timeout
    while any(proc.is_alive() for proc in procs) and time.monotonic() < deadline:
        time.sleep(0.05)
    return [proc for proc in procs if proc.is_alive()]


@pytest.fixture(autouse=True)
def no_leaked_subprocess_workers():
    before = set(_live_children())
    yield
    owned = {
        session._proc.pid
        for session in _sessions_held_by_engine_caches()
        if session._proc is not None
    }
    unaccounted = [
        proc for pid, proc in _live_children().items() if pid not in before and pid not in owned
    ]
    leaked = _still_alive_after(unaccounted, WORKER_EXIT_TIMEOUT_SECONDS)
    if not leaked:
        return
    spawners = {
        session._proc.pid: session for session in list(_all_sessions) if session._proc is not None
    }
    details = []
    for proc in leaked:
        session = spawners.get(proc.pid)
        if session is None:
            origin = "no session owns it"
        else:
            origin = (
                f"spawned by {type(session).__name__}, shutdown requested="
                f"{getattr(session, '_shutdown_requested', 'n/a')}, closed={session._closed}"
            )
        details.append(f"pid {proc.pid} ({proc.name}; {origin})")
        proc.kill()
        proc.join(timeout=5.0)
    pytest.fail("worker process(es) left running after the test: " + "; ".join(details))
