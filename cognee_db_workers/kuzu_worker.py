"""Ladybug (formerly Kuzu) subprocess worker.

Imports only ``ladybug`` + stdlib + harness/protocol. Must not import cognee
— that would defeat the whole point of running the DB client in a separate
process. The module name retains the historical ``kuzu_worker`` spelling so
existing imports (and the ``test_worker_import_hygiene`` allowlist) keep
working without churn.
"""

from __future__ import annotations

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


def _retry_open_locked(ladybug, kwargs, original_exc):
    """Re-attempt ``ladybug.Database(**kwargs)`` after an initial lock-held
    failure, with bounded exponential backoff.

    This is a backstop for the brief window where another worker for the same
    DB path is still releasing its on-disk file lock (e.g. a cache eviction
    whose close is driven by a GC finalizer and so can't be cleanly awaited by
    the creator). The OS frees a dead process's file locks immediately, so once
    the previous worker exits the next attempt succeeds. Only the lock-held
    error is retried; any other ``RuntimeError`` propagates unchanged.

    ``original_exc`` is the first lock-held failure the caller already saw; it is
    re-raised when retries are exhausted — or immediately when retries are
    disabled (``SUBPROCESS_OPEN_LOCK_RETRIES <= 0``), so a misconfigured value
    surfaces the real lock error instead of a spurious ``TypeError``.
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
            # Transient inter-process lock contention with another worker that
            # is still shutting down for the same path — retry with backoff
            # rather than treating it as corruption/migration.
            db = _retry_open_locked(ladybug, req.kwargs, e)
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
