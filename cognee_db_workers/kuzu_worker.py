"""Ladybug (formerly Kuzu) subprocess worker.

Imports only ``ladybug`` + stdlib + harness/protocol. Must not import cognee
— that would defeat the whole point of running the DB client in a separate
process. The module name retains the historical ``kuzu_worker`` spelling so
existing imports (and the ``test_worker_import_hygiene`` allowlist) keep
working without churn.
"""

from __future__ import annotations

import re

from ._kuzu_helpers import install_json_extension_local
from .harness import (
    DEFAULT_DISPATCH,
    HandleRegistry,
    HandleResult,
    Request,
    run_worker_loop,
)
from .kuzu_protocol import (
    OP_CONN_CLOSE,
    OP_CONN_EXECUTE_FETCH_ALL,
    OP_DB_CLOSE,
    OP_DB_INIT,
    OP_INSTALL_JSON,
    OP_LOAD_EXTENSION,
    OP_OPEN_CONNECTION,
    OP_OPEN_DATABASE,
)


_LOCK_HELD_MARKER = "could not set lock on file"

# Ladybug reports the lock holder as "... Lock is held by PID <pid>".
_LOCK_PID_PATTERN = re.compile(r"held by pid\s+(\d+)", re.IGNORECASE)


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a lock-holder PID.

    POSIX-accurate; conservative everywhere else. Returns True whenever the
    process still exists OR liveness can't be determined, so a live or unknown
    holder is never mistaken for a dead one. Only a definitive
    ``ProcessLookupError`` proves the holder is gone.
    """
    import os

    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        # Exists but owned by another user, or the platform can't answer (e.g.
        # signal 0 unsupported on Windows) — assume alive and leave it alone.
        return True
    return True


def _reclaim_stale_lock(db_path, exc) -> bool:
    """Remove a Ladybug lock file orphaned by a dead worker.

    A worker that crashes (OOM, SIGKILL, a hard error mid-open) can leave
    ``<db_path>.lock`` on disk. Every later open then fails with
    "Lock is held by PID <pid>" for a PID that no longer exists, which wedges all
    subsequent ``recall()`` calls (gh #3708). We remove the stale lock only when
    the holder PID is parseable from the error AND confirmed not alive — a live
    or unknown holder is left untouched, so a healthy worker never has its lock
    yanked out from under it.

    Returns True when a stale lock was removed (the caller may retry the open).
    """
    import os

    if not db_path:
        return False
    match = _LOCK_PID_PATTERN.search(str(exc))
    if not match:
        return False
    holder_pid = int(match.group(1))
    if holder_pid == os.getpid() or _pid_alive(holder_pid):
        return False
    try:
        os.remove(str(db_path) + ".lock")
    except (FileNotFoundError, OSError):
        return False
    return True


def _open_locked_with_recovery(ladybug, kwargs, db_path, first_exc):
    """Open a Ladybug database that first failed with a lock-held error.

    Two layers: back off for transient inter-worker contention
    (``_retry_open_locked``), then — if the lock still won't clear — reclaim it
    when it was orphaned by a dead worker (gh #3708) and reopen once. Any
    non-lock error, or a lock whose holder is alive/unknown, propagates unchanged.
    """
    try:
        return _retry_open_locked(ladybug, kwargs, first_exc)
    except RuntimeError as retry_exc:
        if _LOCK_HELD_MARKER not in str(retry_exc).lower():
            raise
        if not _reclaim_stale_lock(db_path, retry_exc):
            raise
        return ladybug.Database(**kwargs)


def _retry_open_locked(ladybug, kwargs, original_exc):
    """Re-attempt ``ladybug.Database(**kwargs)`` after an initial lock-held
    failure, with bounded exponential backoff.

    This is a backstop for the brief window where another worker for the same
    DB path is still releasing its on-disk file lock (e.g. a cache eviction
    whose close is driven by a GC finalizer and so can't be cleanly awaited by
    the creator). The OS frees a dead process's file locks immediately, so once
    the previous worker exits the next attempt succeeds. Only the lock-held
    error is retried; any other ``RuntimeError`` propagates unchanged.

    ``original_exc`` is the first lock-held failure the caller already saw and
    seeds ``last_exc``. When every retry still hits the lock the *last* lock
    error is raised (``last_exc``, updated each attempt); when retries are
    disabled (``SUBPROCESS_OPEN_LOCK_RETRIES <= 0``) the loop never runs so
    ``original_exc`` is raised immediately — either way a real lock error
    surfaces rather than a spurious ``TypeError`` from ``raise None``.
    """
    import time

    from .harness import OPEN_LOCK_BACKOFF, OPEN_LOCK_RETRIES

    last_exc = original_exc
    for attempt in range(OPEN_LOCK_RETRIES):
        # Backoff first — the caller already saw one failure. Cap per-attempt so
        # exponential growth stays bounded (≈ a few seconds total by default).
        time.sleep(min(OPEN_LOCK_BACKOFF * (2**attempt), 0.5))
        try:
            return ladybug.Database(**kwargs)
        except RuntimeError as e:
            if _LOCK_HELD_MARKER not in str(e).lower():
                raise
            last_exc = e
    raise last_exc


def _open_database(registry: HandleRegistry, req: Request) -> HandleResult:
    import ladybug

    try:
        db = ladybug.Database(**req.kwargs)
    except RuntimeError as e:
        db_path = req.kwargs.get("database_path", "")
        message = str(e).lower()

        if _LOCK_HELD_MARKER in message:
            # A lock-held failure is usually transient inter-process contention
            # with another worker still shutting down for the same path (retry
            # with backoff), but it can also be a stale lock orphaned by a dead
            # worker — reclaim that only when its holder PID is confirmed gone
            # (gh #3708), never treating it as corruption/migration.
            db = _open_locked_with_recovery(ladybug, req.kwargs, db_path, e)
        else:
            if "wal" in message:
                # In case of corrupted WAL file preventing database opening, remove the WAL file and try again
                wal_path = db_path + ".wal"
                try:
                    import os

                    os.remove(wal_path)
                except FileNotFoundError:
                    pass
            else:
                from .ladybug_migrate import needs_migration, ladybug_migration

                should_migrate, old_version = needs_migration(db_path, ladybug.__version__)
                if should_migrate:
                    ladybug_migration(
                        new_db=db_path + "_new",
                        old_db=db_path,
                        new_version=ladybug.__version__,
                        old_version=old_version,
                        overwrite=True,
                    )
            db = ladybug.Database(**req.kwargs)

    return HandleResult(value=None, handle_id=registry.register(db))


def _db_init(registry: HandleRegistry, req: Request) -> None:
    db = registry.get(req.handle_id)
    db.init_database()
    return None


def _db_close(registry: HandleRegistry, req: Request) -> None:
    db = registry.pop(req.handle_id)
    if db is not None and hasattr(db, "close"):
        try:
            db.close()
        except Exception:
            pass
    return None


def _open_connection(registry: HandleRegistry, req: Request) -> HandleResult:
    import ladybug

    db_handle_id = req.args[0]
    db = registry.get(db_handle_id)
    conn = ladybug.Connection(db)
    return HandleResult(value=None, handle_id=registry.register(conn))


def _conn_close(registry: HandleRegistry, req: Request) -> None:
    conn = registry.pop(req.handle_id)
    if conn is not None and hasattr(conn, "close"):
        try:
            conn.close()
        except Exception:
            pass
    return None


def _conn_execute_fetch_all(registry: HandleRegistry, req: Request):
    """Run a query and return fully-materialized rows as list[tuple].

    Each cell is unwrapped via ``.as_py()`` when available (Arrow scalars),
    matching the adapter's existing contract.
    """
    conn = registry.get(req.handle_id)
    query = req.args[0]
    params = req.args[1] if len(req.args) > 1 else None

    if params is None:
        result = conn.execute(query)
    else:
        result = conn.execute(query, params)

    rows = []
    try:
        while result.has_next():
            raw = result.get_next()
            rows.append(tuple(cell.as_py() if hasattr(cell, "as_py") else cell for cell in raw))
    finally:
        # Ladybug's Python QueryResult is auto-closed on GC; explicit close is
        # a safety net when available.
        if hasattr(result, "close"):
            try:
                result.close()
            except Exception:
                pass
    return rows


def _install_json(registry: HandleRegistry, req: Request) -> None:
    """Run INSTALL JSON on a throwaway database so the extension is cached."""
    buffer_pool_size = req.args[0] if req.args else 64 * 1024 * 1024
    install_json_extension_local(buffer_pool_size)
    return None


def _load_extension(registry: HandleRegistry, req: Request) -> None:
    conn = registry.get(req.handle_id)
    extension_name = req.args[0]
    try:
        conn.execute(f"LOAD EXTENSION {extension_name};")
    except RuntimeError as error:
        if "not been installed" not in str(error):
            raise
        # The warm-up INSTALL on the throwaway database is best-effort and
        # can fail silently (e.g. a transient network error downloading the
        # extension on a fresh machine). Install on the live connection and
        # retry once; if INSTALL fails here it raises with the real cause.
        conn.execute(f"INSTALL {extension_name};")
        conn.execute(f"LOAD EXTENSION {extension_name};")
    return None


DISPATCH = {
    **DEFAULT_DISPATCH,
    OP_OPEN_DATABASE: _open_database,
    OP_DB_INIT: _db_init,
    OP_DB_CLOSE: _db_close,
    OP_OPEN_CONNECTION: _open_connection,
    OP_CONN_CLOSE: _conn_close,
    OP_CONN_EXECUTE_FETCH_ALL: _conn_execute_fetch_all,
    OP_INSTALL_JSON: _install_json,
    OP_LOAD_EXTENSION: _load_extension,
}


def worker_main(req_q, resp_q) -> None:
    """Entry point passed to ``mp.Process(target=...)``.

    Must be a top-level function so it can be pickled by spawn.
    """
    run_worker_loop(DISPATCH, req_q, resp_q)
