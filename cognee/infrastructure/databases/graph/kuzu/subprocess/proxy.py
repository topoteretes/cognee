"""Main-process proxies that look like ``kuzu.Database`` / ``kuzu.Connection``
but route every call through a dedicated subprocess that only imports ``kuzu``.

The subprocess machinery lives in the top-level ``cognee_db_workers`` package
so the child never imports cognee.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Optional

from cognee_db_workers.harness import (
    ReplayStep,
    Request,
    SubprocessSession,
    get_process_rss_bytes,
    spawn_without_main,
)
from cognee_db_workers.kuzu_protocol import (
    OP_CONN_CLOSE,
    OP_CONN_EXECUTE_FETCH_ALL,
    OP_DB_CLOSE,
    OP_DB_INIT,
    OP_INSTALL_JSON,
    OP_LOAD_EXTENSION,
    OP_OPEN_CONNECTION,
    OP_OPEN_DATABASE,
)
from cognee_db_workers.kuzu_worker import worker_main


class KuzuSubprocessSession(SubprocessSession):
    """A ``SubprocessSession`` specialized for Kuzu. Owns the spawned process
    and presents a uniform ``.call`` / ``.call_async`` interface.
    """

    @classmethod
    def start(cls, *, max_retries: int = 2) -> "KuzuSubprocessSession":
        ctx = mp.get_context("spawn")

        def _spawn():
            req_q = ctx.Queue()
            resp_q = ctx.Queue()
            proc = ctx.Process(
                target=worker_main,
                args=(req_q, resp_q),
                daemon=True,
            )
            with spawn_without_main():
                proc.start()
            return proc, req_q, resp_q

        proc, req_q, resp_q = _spawn()
        session = cls(
            proc, req_q, resp_q,
            respawn_factory=_spawn,
            max_retries=max_retries,
        )
        try:
            session.wait_for_ready()
        except Exception:
            session.shutdown(timeout=2.0)
            raise
        return session

    def memory_used_bytes(self) -> int:
        if self._proc.pid is None:
            return 0
        return get_process_rss_bytes(self._proc.pid)


class _Materialized:
    """In-memory facsimile of a kuzu ``QueryResult`` backed by already-fetched rows.

    The adapter iterates results as ``has_next()`` / ``get_next()`` returning
    tuples of Python scalars. We preserve that contract so callers don't need
    to change.
    """

    def __init__(self, rows: list):
        self._rows = rows
        self._i = 0

    def has_next(self) -> bool:
        return self._i < len(self._rows)

    def get_next(self):
        row = self._rows[self._i]
        self._i += 1
        return row

    def close(self) -> None:  # parity with kuzu.QueryResult
        self._rows = []
        self._i = 0


class RemoteKuzuDatabase:
    """Main-side proxy for ``kuzu.Database``.

    Registers replay steps with the session so that after a worker respawn
    the database is reopened with the same kwargs and its handle id is
    rebound in place. ``init_database`` is also replayed.
    """

    def __init__(
        self,
        session: KuzuSubprocessSession,
        *,
        db_path: str,
        buffer_pool_size: int,
        max_num_threads: int,
        max_db_size: int,
    ) -> None:
        self._session = session
        self._handle_id: Optional[int] = None
        self._db_path = db_path
        self._open_kwargs = dict(
            database_path=db_path,
            buffer_pool_size=buffer_pool_size,
            max_num_threads=max_num_threads,
            max_db_size=max_db_size,
        )
        self._initialized = False
        self._open()
        self._register_replay()

    @property
    def handle_id(self) -> int:
        assert self._handle_id is not None, "database handle closed"
        return self._handle_id

    def _open(self) -> None:
        resp = self._session.call(Request(op=OP_OPEN_DATABASE, kwargs=self._open_kwargs))
        self._handle_id = resp.new_handle_id

    def _apply_new_db_handle(self, new_hid: int) -> Optional[int]:
        old = self._handle_id
        self._handle_id = new_hid
        return old

    def _register_replay(self) -> None:
        """Re-open the database with the same kwargs after a respawn."""
        self._session.add_replay_step(
            ReplayStep(
                make_request=lambda: Request(op=OP_OPEN_DATABASE, kwargs=self._open_kwargs),
                apply_new_handle=self._apply_new_db_handle,
            )
        )

    def init_database(self) -> None:
        self._session.call(Request(op=OP_DB_INIT, handle_id=self._handle_id))
        self._initialized = True
        # After the first init, replay it too — lambda reads the current
        # (possibly remapped) handle_id at replay time.
        self._session.add_replay_step(
            ReplayStep(
                make_request=lambda: Request(op=OP_DB_INIT, handle_id=self._handle_id),
            )
        )

    def close(self) -> None:
        if self._handle_id is None:
            return
        try:
            self._session.call(Request(op=OP_DB_CLOSE, handle_id=self._handle_id))
        finally:
            self._handle_id = None


class RemoteKuzuConnection:
    """Main-side proxy for ``kuzu.Connection``. Mirrors the tiny slice of the
    kuzu API the adapter actually uses: ``execute(query, params)`` + ``close()``.

    Registers a replay step that re-opens the connection against the Database
    proxy's (possibly remapped) handle after a respawn, and appends a replay
    step per ``load_extension`` call so they're reloaded on the new
    connection too.
    """

    def __init__(self, session: KuzuSubprocessSession, database: RemoteKuzuDatabase) -> None:
        self._session = session
        self._database = database
        resp = session.call(
            Request(op=OP_OPEN_CONNECTION, args=(database.handle_id,))
        )
        self._handle_id: Optional[int] = resp.new_handle_id
        self._register_replay()

    @property
    def handle_id(self) -> int:
        assert self._handle_id is not None, "connection handle closed"
        return self._handle_id

    def _apply_new_conn_handle(self, new_hid: int) -> Optional[int]:
        old = self._handle_id
        self._handle_id = new_hid
        return old

    def _register_replay(self) -> None:
        self._session.add_replay_step(
            ReplayStep(
                make_request=lambda: Request(
                    op=OP_OPEN_CONNECTION, args=(self._database.handle_id,)
                ),
                apply_new_handle=self._apply_new_conn_handle,
            )
        )

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> _Materialized:
        """Execute a query; return a ``QueryResult``-like iterator of fully
        materialized rows.
        """
        resp = self._session.call(
            Request(
                op=OP_CONN_EXECUTE_FETCH_ALL,
                handle_id=self._handle_id,
                args=(query, params),
            )
        )
        rows = resp.result or []
        return _Materialized(rows)

    def load_extension(self, name: str) -> None:
        self._session.call(
            Request(op=OP_LOAD_EXTENSION, handle_id=self._handle_id, args=(name,))
        )
        # Replay the same extension load on any fresh connection.
        self._session.add_replay_step(
            ReplayStep(
                make_request=lambda name=name: Request(
                    op=OP_LOAD_EXTENSION,
                    handle_id=self._handle_id,
                    args=(name,),
                ),
            )
        )

    def close(self) -> None:
        if self._handle_id is None:
            return
        try:
            self._session.call(Request(op=OP_CONN_CLOSE, handle_id=self._handle_id))
        finally:
            self._handle_id = None


def install_json_extension(session: KuzuSubprocessSession, buffer_pool_size: int) -> None:
    """Run INSTALL JSON on a throwaway database inside the worker.

    Also registered as a replay step so a respawned worker gets the same
    setup. INSTALL JSON is idempotent at the Kuzu level — extensions are
    cached in the user's Kuzu state dir — so re-running on a fresh worker is
    cheap and safe.
    """
    session.call(Request(op=OP_INSTALL_JSON, args=(buffer_pool_size,)))
    session.add_replay_step(
        ReplayStep(
            make_request=lambda: Request(op=OP_INSTALL_JSON, args=(buffer_pool_size,)),
        )
    )
