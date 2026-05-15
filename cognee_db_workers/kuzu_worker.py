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


def _open_database(registry: HandleRegistry, req: Request) -> HandleResult:
    import ladybug

    try:
        db = ladybug.Database(**req.kwargs)
    except RuntimeError as e:
        db_path = req.kwargs.get("database_path", "")

        if "wal" in str(e).lower():
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
