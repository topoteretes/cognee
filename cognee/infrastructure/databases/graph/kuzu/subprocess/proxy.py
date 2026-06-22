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
    def start(cls, *, max_retries: Optional[int] = None) -> "KuzuSubprocessSession":
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
        # Forward ``max_retries`` only when explicitly set so the
        # ``SUBPROCESS_MAX_RETRIES`` env var (read by
        # ``SubprocessSession.__init__``) takes effect when the caller
        # doesn't override.
        kwargs: Dict[str, Any] = {"respawn_factory": _spawn}
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        session = cls(proc, req_q, resp_q, **kwargs)
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
        # Tracks replay steps THIS proxy registered with the session, so
        # ``close()`` can deregister them. Without this, a worker respawn
        # after ``close()`` would replay the OPEN step and silently
        # resurrect a handle the user already closed.
        self._replay_steps: list[ReplayStep] = []
        self._open()
        self._register_replay()

    @property
    def handle_id(self) -> int:
        if self._handle_id is None:
            raise RuntimeError("database handle closed")
        return self._handle_id

    def _open(self) -> None:
        resp = self._session.call(Request(op=OP_OPEN_DATABASE, kwargs=self._open_kwargs))
        self._handle_id = resp.new_handle_id

    def _apply_new_db_handle(self, new_hid: int) -> Optional[int]:
        old = self._handle_id
        if old is None:
            # Defense-in-depth mirror of the LanceDB ``_apply_new_handle``
            # guard. The current ``close()`` ordering (deregister → call →
            # clear in finally) avoids the race today, but anyone changing
            # that ordering shouldn't silently regress to resurrecting a
            # closed proxy.
            return None
        self._handle_id = new_hid
        return old

    def _register_replay(self) -> None:
        """Re-open the database with the same kwargs after a respawn."""
        step = ReplayStep(
            make_request=lambda: Request(op=OP_OPEN_DATABASE, kwargs=self._open_kwargs),
            apply_new_handle=self._apply_new_db_handle,
        )
        self._session.add_replay_step(step)
        self._replay_steps.append(step)

    def init_database(self) -> None:
        # Use the validated property so a use-after-close fails locally with
        # ``RuntimeError("database handle closed")`` rather than sending
        # ``handle_id=None`` to the worker, which would surface as a
        # confusing protocol error far from the bug.
        self._session.call(Request(op=OP_DB_INIT, handle_id=self.handle_id))
        self._initialized = True
        # After the first init, replay it too — lambda reads the current
        # (possibly remapped) handle_id at replay time.
        step = ReplayStep(
            make_request=lambda: Request(op=OP_DB_INIT, handle_id=self._handle_id),
        )
        self._session.add_replay_step(step)
        self._replay_steps.append(step)

    def close(self) -> None:
        if self._handle_id is None:
            return
        # Deregister BEFORE the close call: if the worker dies between
        # here and ``OP_DB_CLOSE`` returning, the next ``_respawn`` must
        # NOT replay the OPEN step and resurrect a handle the user just
        # closed. Doing it after the close would also work for the
        # happy path, but only deregister-first is race-safe.
        for step in self._replay_steps:
            self._session.remove_replay_step(step)
        self._replay_steps.clear()
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
        resp = session.call(Request(op=OP_OPEN_CONNECTION, args=(database.handle_id,)))
        self._handle_id: Optional[int] = resp.new_handle_id
        # Tracks replay steps THIS proxy registered (one for the
        # connection itself + one per loaded extension). ``close()``
        # deregisters them all so a post-close respawn doesn't reopen a
        # connection the user already discarded.
        self._replay_steps: list[ReplayStep] = []
        self._register_replay()

    @property
    def handle_id(self) -> int:
        if self._handle_id is None:
            raise RuntimeError("connection handle closed")
        return self._handle_id

    def _apply_new_conn_handle(self, new_hid: int) -> Optional[int]:
        old = self._handle_id
        if old is None:
            # See ``_apply_new_db_handle`` for the rationale — same guard.
            return None
        self._handle_id = new_hid
        return old

    def _register_replay(self) -> None:
        step = ReplayStep(
            make_request=lambda: Request(op=OP_OPEN_CONNECTION, args=(self._database.handle_id,)),
            apply_new_handle=self._apply_new_conn_handle,
        )
        self._session.add_replay_step(step)
        self._replay_steps.append(step)

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> _Materialized:
        """Execute a query; return a ``QueryResult``-like iterator of fully
        materialized rows.
        """
        # ``self.handle_id`` (property) raises if the connection was closed,
        # so a use-after-close fails locally instead of sending
        # ``handle_id=None`` to the worker.
        resp = self._session.call(
            Request(
                op=OP_CONN_EXECUTE_FETCH_ALL,
                handle_id=self.handle_id,
                args=(query, params),
            )
        )
        rows = resp.result or []
        return _Materialized(rows)

    def load_extension(self, name: str) -> None:
        self._session.call(Request(op=OP_LOAD_EXTENSION, handle_id=self.handle_id, args=(name,)))
        # Replay the same extension load on any fresh connection.
        step = ReplayStep(
            make_request=lambda name=name: Request(
                op=OP_LOAD_EXTENSION,
                handle_id=self._handle_id,
                args=(name,),
            ),
        )
        self._session.add_replay_step(step)
        self._replay_steps.append(step)

    def close(self) -> None:
        if self._handle_id is None:
            return
        # Deregister replay steps BEFORE the close call so a worker
        # respawn between here and ``OP_CONN_CLOSE`` returning can't
        # replay the OPEN_CONNECTION / LOAD_EXTENSION steps and
        # resurrect a connection the user already closed.
        for step in self._replay_steps:
            self._session.remove_replay_step(step)
        self._replay_steps.clear()
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
