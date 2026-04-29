"""Kuzu subprocess worker. Imports only ``kuzu`` + stdlib + harness/protocol.

Must not import cognee — that would defeat the whole point of running the DB
client in a separate process.
"""

from __future__ import annotations

import tempfile

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
    import kuzu

    db = kuzu.Database(**req.kwargs)
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
    import kuzu

    db_handle_id = req.args[0]
    db = registry.get(db_handle_id)
    conn = kuzu.Connection(db)
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
            rows.append(
                tuple(cell.as_py() if hasattr(cell, "as_py") else cell for cell in raw)
            )
    finally:
        # Kuzu's Python QueryResult is auto-closed on GC; explicit close is a
        # safety net when available.
        if hasattr(result, "close"):
            try:
                result.close()
            except Exception:
                pass
    return rows


def _install_json(registry: HandleRegistry, req: Request) -> None:
    """Run INSTALL JSON on a throwaway database so the extension is cached.
    Matches the adapter's existing ``_install_json_extension`` behavior.
    """
    import kuzu

    buffer_pool_size = req.args[0] if req.args else 64 * 1024 * 1024

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tmp:
        temp_db_path = tmp.name
        try:
            tmp_db = kuzu.Database(
                temp_db_path,
                buffer_pool_size=buffer_pool_size,
            )
            tmp_db.init_database()
            conn = kuzu.Connection(tmp_db)
            try:
                conn.execute("INSTALL JSON;")
            except Exception:
                # Already installed / unavailable — parity with the current
                # adapter behavior, which also swallows this.
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    tmp_db.close()
                except Exception:
                    pass
        except Exception:
            # Best-effort install; swallowing matches the current adapter.
            pass
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
