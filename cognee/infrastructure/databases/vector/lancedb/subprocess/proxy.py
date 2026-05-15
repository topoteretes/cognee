"""Main-process proxies that look like the slice of ``lancedb.AsyncConnection``
and table/builder types the cognee adapter uses, but route every call through
a subprocess that only imports ``lancedb`` + ``pyarrow`` (no cognee).
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa

from cognee_db_workers.harness import (
    ReplayStep,
    Request,
    SubprocessSession,
    get_process_rss_bytes,
    spawn_without_main,
)
from cognee_db_workers.lancedb_protocol import (
    OP_CONNECT,
    OP_CREATE_TABLE,
    OP_DROP_TABLE,
    OP_OPEN_TABLE,
    OP_TABLE_ADD,
    OP_TABLE_COUNT_ROWS,
    OP_TABLE_DELETE,
    OP_TABLE_MERGE_INSERT_EXECUTE,
    OP_TABLE_NAMES,
    OP_TABLE_QUERY_EXECUTE,
    OP_TABLE_RELEASE,
    OP_TABLE_TO_ARROW,
    OP_TABLE_VECTOR_SEARCH_EXECUTE,
)
from cognee_db_workers.lancedb_worker import worker_main


class LanceDBSubprocessSession(SubprocessSession):
    @classmethod
    def start(cls, *, max_retries: Optional[int] = None) -> "LanceDBSubprocessSession":
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


class _BuilderChain:
    """Accumulates fluent-builder calls on the main side; dispatches a single
    RPC on the terminal method.
    """

    def __init__(
        self,
        session: LanceDBSubprocessSession,
        table_handle_id: int,
        op_code: int,
        root_args: tuple,
    ):
        self._session = session
        self._table_handle_id = table_handle_id
        self._op_code = op_code
        self._root_args = root_args
        self._steps: List[Tuple[str, tuple, dict]] = []

    def _add(self, name: str, *args, **kwargs) -> "_BuilderChain":
        self._steps.append((name, args, kwargs))
        return self

    async def _terminal(self, terminal_name: str, *args, **kwargs):
        resp = await self._session.call_async(
            Request(
                op=self._op_code,
                handle_id=self._table_handle_id,
                args=(self._root_args, list(self._steps), terminal_name, args, kwargs),
            )
        )
        return resp.result


class RemoteQuery(_BuilderChain):
    """Proxy for ``Table.query()``."""

    def where(self, predicate: str) -> "RemoteQuery":
        return self._add("where", predicate)

    async def to_list(self) -> list:
        return await self._terminal("to_list")


class RemoteVectorSearch(_BuilderChain):
    """Proxy for ``Table.vector_search(vector)``."""

    def distance_type(self, metric: str) -> "RemoteVectorSearch":
        return self._add("distance_type", metric)

    def where(self, predicate: str) -> "RemoteVectorSearch":
        return self._add("where", predicate)

    def select(self, columns: List[str]) -> "RemoteVectorSearch":
        return self._add("select", columns)

    def limit(self, n: int) -> "RemoteVectorSearch":
        return self._add("limit", n)

    async def to_list(self) -> list:
        return await self._terminal("to_list")


class RemoteMergeInsert(_BuilderChain):
    """Proxy for ``Table.merge_insert(key)``."""

    def when_matched_update_all(self) -> "RemoteMergeInsert":
        return self._add("when_matched_update_all")

    def when_not_matched_insert_all(self) -> "RemoteMergeInsert":
        return self._add("when_not_matched_insert_all")

    async def execute(self, records: list) -> None:
        await self._terminal("execute", records)


class RemoteLanceDBTable:
    """Main-side proxy for an open ``lancedb`` table.

    Holds a worker-side table handle that must be released when the proxy is
    dropped — otherwise the worker's ``HandleRegistry`` grows unboundedly as
    cognee calls ``get_collection`` (which opens a table) on every operation.
    """

    def __init__(self, session: LanceDBSubprocessSession, handle_id: int, name: str):
        self._session = session
        self._handle_id: Optional[int] = handle_id
        self.name = name
        # Register a replay step so that if the worker dies and respawns, the
        # table is re-opened and the handle remap rewrites in-flight Requests
        # from the old handle id to the new one. Without this, a retried
        # ``table.add(...)`` after a worker crash would send the stale
        # handle id and fail with "unknown handle".
        self._replay_step = ReplayStep(
            make_request=lambda: Request(op=OP_OPEN_TABLE, args=(self.name,)),
            apply_new_handle=self._apply_new_handle,
        )
        self._session.add_replay_step(self._replay_step)

    def _apply_new_handle(self, new_handle_id: int) -> Optional[int]:
        """Called by the session after a successful replay of our OPEN_TABLE
        step. Returns the previous handle id so the session's handle-remap
        dict rewrites stale in-flight Requests. Returns ``None`` if the proxy
        has already been released (no remap needed in that case).
        """
        old = self._handle_id
        if old is None:
            # Race window: ``release()`` cleared ``_handle_id`` and
            # deregistered our step (without ``_rpc_lock``) AFTER
            # ``_respawn`` already snapshotted ``_replay_steps``. Without
            # this guard, ``self._handle_id = new_handle_id`` would
            # resurrect a table the user already released. The new
            # worker-side handle that replay created is orphaned (one
            # leaked handle per race occurrence, bounded by respawn
            # count); freeing it would require session-level cleanup
            # which is out of scope here.
            return None
        self._handle_id = new_handle_id
        return old

    @property
    def handle_id(self) -> int:
        if self._handle_id is None:
            raise RuntimeError("lancedb table handle released")
        return self._handle_id

    def _deregister_replay(self) -> None:
        step = getattr(self, "_replay_step", None)
        if step is not None:
            try:
                self._session.remove_replay_step(step)
            except Exception:
                pass
            self._replay_step = None

    async def release(self) -> None:
        """Release the worker-side table handle. Idempotent."""
        if self._handle_id is None:
            return
        hid = self._handle_id
        self._handle_id = None
        self._deregister_replay()
        try:
            await self._session.call_async(Request(op=OP_TABLE_RELEASE, handle_id=hid))
        except Exception:
            # Session already torn down; the handle dies with the worker.
            pass

    def release_sync(self) -> None:
        """Sync variant for use in ``__del__`` / non-async contexts."""
        if self._handle_id is None:
            return
        hid = self._handle_id
        self._handle_id = None
        self._deregister_replay()
        try:
            self._session.call(Request(op=OP_TABLE_RELEASE, handle_id=hid))
        except Exception:
            pass

    def __del__(self):
        # Best-effort; async release is preferred. Only try sync release if the
        # session still looks alive — otherwise the handle is already gone.
        try:
            if self._handle_id is not None and not self._session._closed:
                self.release_sync()
        except Exception:
            pass

    async def count_rows(self) -> int:
        resp = await self._session.call_async(
            Request(op=OP_TABLE_COUNT_ROWS, handle_id=self.handle_id)
        )
        return int(resp.result)

    async def to_arrow(self) -> pa.Table:
        resp = await self._session.call_async(
            Request(op=OP_TABLE_TO_ARROW, handle_id=self.handle_id)
        )
        buf = resp.result
        # Use the reader as a context manager so its native buffers are
        # released as soon as ``read_all()`` materializes the Table —
        # otherwise the reader (and the wrapped pa.py_buffer) lingers
        # until GC and accumulates across repeated calls.
        with pa.ipc.open_stream(pa.py_buffer(buf)) as reader:
            return reader.read_all()

    async def add(self, records: list) -> None:
        await self._session.call_async(
            Request(op=OP_TABLE_ADD, handle_id=self.handle_id, args=(records,))
        )

    async def delete(self, where_expr: str) -> None:
        await self._session.call_async(
            Request(op=OP_TABLE_DELETE, handle_id=self.handle_id, args=(where_expr,))
        )

    def query(self) -> RemoteQuery:
        return RemoteQuery(self._session, self.handle_id, OP_TABLE_QUERY_EXECUTE, ())

    def vector_search(self, vector: list) -> RemoteVectorSearch:
        return RemoteVectorSearch(
            self._session, self.handle_id, OP_TABLE_VECTOR_SEARCH_EXECUTE, (vector,)
        )

    def merge_insert(self, key: str) -> RemoteMergeInsert:
        return RemoteMergeInsert(
            self._session, self.handle_id, OP_TABLE_MERGE_INSERT_EXECUTE, (key,)
        )

    # Context-manager protocol. Upstream ``lancedb.AsyncTable`` only defines
    # the sync ``__enter__`` / ``__exit__`` pair (its ``__exit__`` calls
    # ``close()``); we additionally support ``async with`` since the proxy is
    # async-first and the async release is the cheaper path under an event
    # loop. Both exits release the worker-side handle and deregister the
    # replay step — using the table after exit raises ``RuntimeError`` from
    # the ``handle_id`` property, matching upstream's
    # "any attempt to use the table after it has been closed will raise".
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_sync()
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
        return False


class RemoteLanceDBConnection:
    """Main-side proxy for ``lancedb.AsyncConnection``. Only the subset of the
    API used by cognee's LanceDBAdapter is exposed.
    """

    def __init__(self, session: LanceDBSubprocessSession, url: str, api_key: Optional[str]):
        self._session = session
        self._url = url
        self._api_key = api_key
        self._connected = False
        # Serializes first-time connects so concurrent callers (e.g. via
        # ``asyncio.gather`` over multiple adapter operations) don't both
        # observe ``_connected == False``, both issue ``OP_CONNECT``, and
        # both append a replay step. Without this guard, ``_replay_steps``
        # accumulates one duplicate per racing caller and every respawn
        # then reconnects N times. ``asyncio.Lock`` binds to the running
        # loop on first ``async with``, so eager construction here is fine.
        self._connect_lock = asyncio.Lock()
        self._connect_replay_step: Optional[ReplayStep] = None

    async def connect(self) -> None:
        # Fast path: already connected, no lock needed.
        if self._connected:
            return
        # Slow path: re-check under the lock so only one task does the
        # real connect + replay-step registration.
        async with self._connect_lock:
            if self._connected:
                return
            await self._session.call_async(
                Request(op=OP_CONNECT, kwargs={"url": self._url, "api_key": self._api_key})
            )
            self._connected = True
            # Replay step: on worker respawn, re-establish the underlying
            # lancedb connection before any other op fires. Registered
            # exactly once — the ``is None`` guard makes ``connect()``
            # idempotent against future flows that flip ``_connected``
            # back to False (none today, but defensive).
            if self._connect_replay_step is None:
                self._connect_replay_step = ReplayStep(
                    make_request=lambda: Request(
                        op=OP_CONNECT,
                        kwargs={"url": self._url, "api_key": self._api_key},
                    ),
                )
                self._session.add_replay_step(self._connect_replay_step)

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def table_names(self) -> list:
        await self._ensure_connected()
        resp = await self._session.call_async(Request(op=OP_TABLE_NAMES))
        return list(resp.result or [])

    async def create_table(
        self, name: str, schema: pa.Schema, exist_ok: bool = True
    ) -> "RemoteLanceDBTable":
        await self._ensure_connected()
        # Use Arrow's native IPC schema serialization rather than pickle —
        # the worker is otherwise an unconditional ``pickle.loads`` target
        # for any value that lands on its request queue. ``ipc`` is a
        # typed, structured format and ``read_schema`` rejects anything
        # that isn't a valid Arrow schema.
        schema_bytes = schema.serialize().to_pybytes()
        await self._session.call_async(
            Request(op=OP_CREATE_TABLE, args=(name, schema_bytes, exist_ok))
        )
        return await self.open_table(name)

    async def open_table(self, name: str) -> "RemoteLanceDBTable":
        await self._ensure_connected()
        resp = await self._session.call_async(Request(op=OP_OPEN_TABLE, args=(name,)))
        return RemoteLanceDBTable(self._session, resp.new_handle_id, name)

    async def drop_table(self, name: str) -> None:
        await self._ensure_connected()
        await self._session.call_async(Request(op=OP_DROP_TABLE, args=(name,)))
