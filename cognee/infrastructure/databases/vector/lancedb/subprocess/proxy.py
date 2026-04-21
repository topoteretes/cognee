"""Main-process proxies that look like the slice of ``lancedb.AsyncConnection``
and table/builder types the cognee adapter uses, but route every call through
a subprocess that only imports ``lancedb`` + ``pyarrow`` (no cognee).
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from typing import Any, List, Optional, Tuple

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
    def start(cls, *, max_retries: int = 2) -> "LanceDBSubprocessSession":
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


class _BuilderChain:
    """Accumulates fluent-builder calls on the main side; dispatches a single
    RPC on the terminal method.
    """

    def __init__(self, session: LanceDBSubprocessSession, table_handle_id: int,
                 op_code: int, root_args: tuple):
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

    @property
    def handle_id(self) -> int:
        assert self._handle_id is not None, "lancedb table handle released"
        return self._handle_id

    async def release(self) -> None:
        """Release the worker-side table handle. Idempotent."""
        if self._handle_id is None:
            return
        hid = self._handle_id
        self._handle_id = None
        try:
            await self._session.call_async(
                Request(op=OP_TABLE_RELEASE, handle_id=hid)
            )
        except Exception:
            # Session already torn down; the handle dies with the worker.
            pass

    def release_sync(self) -> None:
        """Sync variant for use in ``__del__`` / non-async contexts."""
        if self._handle_id is None:
            return
        hid = self._handle_id
        self._handle_id = None
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
            Request(op=OP_TABLE_COUNT_ROWS, handle_id=self._handle_id)
        )
        return int(resp.result)

    async def to_arrow(self) -> pa.Table:
        resp = await self._session.call_async(
            Request(op=OP_TABLE_TO_ARROW, handle_id=self._handle_id)
        )
        buf = resp.result
        reader = pa.ipc.open_stream(pa.py_buffer(buf))
        return reader.read_all()

    async def add(self, records: list) -> None:
        await self._session.call_async(
            Request(op=OP_TABLE_ADD, handle_id=self._handle_id, args=(records,))
        )

    async def delete(self, where_expr: str) -> None:
        await self._session.call_async(
            Request(op=OP_TABLE_DELETE, handle_id=self._handle_id, args=(where_expr,))
        )

    def query(self) -> RemoteQuery:
        return RemoteQuery(self._session, self._handle_id, OP_TABLE_QUERY_EXECUTE, ())

    def vector_search(self, vector: list) -> RemoteVectorSearch:
        return RemoteVectorSearch(
            self._session, self._handle_id, OP_TABLE_VECTOR_SEARCH_EXECUTE, (vector,)
        )

    def merge_insert(self, key: str) -> RemoteMergeInsert:
        return RemoteMergeInsert(
            self._session, self._handle_id, OP_TABLE_MERGE_INSERT_EXECUTE, (key,)
        )

    # Parity with lancedb's context manager protocol (used as `with table:` in
    # older adapter snippets). Both enter/exit are no-ops here.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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

    async def connect(self) -> None:
        if self._connected:
            return
        await self._session.call_async(
            Request(op=OP_CONNECT, kwargs={"url": self._url, "api_key": self._api_key})
        )
        self._connected = True
        # Replay step: on worker respawn, re-establish the underlying lancedb
        # connection before any other op fires. Registered exactly once,
        # after the first successful connect.
        self._session.add_replay_step(
            ReplayStep(
                make_request=lambda: Request(
                    op=OP_CONNECT,
                    kwargs={"url": self._url, "api_key": self._api_key},
                ),
            )
        )

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def table_names(self) -> list:
        await self._ensure_connected()
        resp = await self._session.call_async(Request(op=OP_TABLE_NAMES))
        return list(resp.result or [])

    async def create_table(self, name: str, schema: pa.Schema, exist_ok: bool = True) -> "RemoteLanceDBTable":
        await self._ensure_connected()
        schema_bytes = pickle.dumps(schema)
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
