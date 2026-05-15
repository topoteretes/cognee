"""Adapter for Ladybug graph database."""

import os
import json
import asyncio
import threading
import tempfile
from uuid import UUID, uuid5, NAMESPACE_OID
from ladybug import Connection
from ladybug.database import Database
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Union, Optional, Tuple, Type, Set
from cognee.modules.observability import OtelStatusCode as StatusCode
from cognee.exceptions import CogneeValidationError
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.utils.run_sync import run_sync
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.engine.utils.generate_timestamp_datapoint import date_to_int
from cognee.tasks.temporal_graph.models import Timestamp
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.modules.observability import new_span
from cognee.modules.observability.tracing import (
    COGNEE_DB_SYSTEM,
    COGNEE_DB_QUERY,
    COGNEE_DB_ROW_COUNT,
    redact_secrets,
)

logger = get_logger()

DEFAULT_KUZU_BUFFER_POOL_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_KUZU_MAX_DB_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB


cache_config = get_cache_config()
if cache_config.shared_ladybug_lock:
    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine


class LadybugAdapter(GraphDBInterface):
    """
    Adapter for Ladybug graph database operations with improved consistency and async support.

    This class facilitates operations for working with the Ladybug graph database, supporting
    both direct database queries and a structured asynchronous interface for node and edge
    management. It contains methods for querying, adding, and deleting nodes and edges as
    well as for graph metrics and data extraction.
    """

    @classmethod
    def create_subprocess(
        cls,
        db_path: str,
        kuzu_num_threads: int = 0,
        kuzu_buffer_pool_size: int = DEFAULT_KUZU_BUFFER_POOL_SIZE,
        kuzu_max_db_size: int = DEFAULT_KUZU_MAX_DB_SIZE,
    ) -> "LadybugAdapter":
        """Create a LadybugAdapter running in subprocess-proxy mode."""
        db_parent = os.path.dirname(os.path.abspath(db_path))
        if db_parent:
            os.makedirs(db_parent, exist_ok=True)

        from cognee.infrastructure.databases.graph.kuzu.subprocess.proxy import (
            KuzuSubprocessSession,
            RemoteKuzuConnection,
            RemoteKuzuDatabase,
            install_json_extension,
        )

        session = KuzuSubprocessSession.start()
        try:
            install_json_extension(session, kuzu_buffer_pool_size)

            if cache_config.shared_ladybug_lock:
                # Don't open persistent handles — the per-query Redis lock
                # path will open/close them via _rebuild_subprocess_proxies
                # and _drop_native_resources on each query.
                return cls(
                    db_path=db_path,
                    kuzu_num_threads=kuzu_num_threads,
                    kuzu_buffer_pool_size=kuzu_buffer_pool_size,
                    kuzu_max_db_size=kuzu_max_db_size,
                    subprocess_mode=True,
                    session=session,
                )

            db = RemoteKuzuDatabase(
                session,
                db_path=db_path,
                buffer_pool_size=kuzu_buffer_pool_size,
                max_num_threads=kuzu_num_threads,
                max_db_size=kuzu_max_db_size,
            )
            db.init_database()
            conn = RemoteKuzuConnection(session, db)
            conn.load_extension("JSON")

            return cls(
                db_path=db_path,
                kuzu_num_threads=kuzu_num_threads,
                kuzu_buffer_pool_size=kuzu_buffer_pool_size,
                kuzu_max_db_size=kuzu_max_db_size,
                subprocess_mode=True,
                database=db,
                connection=conn,
                session=session,
            )
        except Exception:
            session.shutdown(timeout=2.0)
            raise

    def __init__(
        self,
        db_path: str,
        kuzu_num_threads: int = 0,
        kuzu_buffer_pool_size: int = DEFAULT_KUZU_BUFFER_POOL_SIZE,
        kuzu_max_db_size: int = DEFAULT_KUZU_MAX_DB_SIZE,
        *,
        subprocess_mode: bool = False,
        database: Optional[Any] = None,
        connection: Optional[Any] = None,
        session: Optional[Any] = None,
    ):
        """Initialize Ladybug database connection and schema.

        Parameters
        ----------
        db_path:
            Path to the Kuzu database directory.
        kuzu_num_threads:
            Maximum number of threads Kuzu uses to execute queries. ``0`` keeps
            Kuzu's internal default (one per CPU).
        kuzu_buffer_pool_size:
            Maximum size of the Kuzu buffer pool in bytes.
        kuzu_max_db_size:
            Maximum on-disk database size in bytes. Configurable via the
            ``KUZU_MAX_DB_SIZE`` env var (see ``GraphConfig``); some users
            need this above the default 4 GB for large graphs.
        subprocess_mode:
            When True, the adapter runs in subprocess-proxy mode: the
            native ladybug.Database/Connection live in a dedicated worker
            process. Requires ``session``. When ``shared_ladybug_lock`` is
            disabled, ``database`` and ``connection`` must also be provided
            (persistent handles). When ``shared_ladybug_lock`` is enabled,
            handles are opened/closed per query via the Redis lock path,
            so ``database`` and ``connection`` are left None.
        database, connection:
            Pre-built Database/Connection proxies for the subprocess worker.
        session:
            The subprocess session that owns the worker process. After a
            transient drop (e.g. ``delete_graph``) the adapter rebuilds
            proxies lazily against the surviving session; after ``close()``
            the session is zeroed and the adapter is in a permanent error
            state.
        """
        if subprocess_mode:
            if session is None:
                raise ValueError("subprocess_mode requires a session.")
            if not cache_config.shared_ladybug_lock and (database is None or connection is None):
                raise ValueError(
                    "subprocess_mode without shared_ladybug_lock requires database and connection."
                )
        self.open_connections = 0
        self.db_path = db_path
        self.kuzu_num_threads = kuzu_num_threads
        self.kuzu_buffer_pool_size = kuzu_buffer_pool_size
        self.kuzu_max_db_size = kuzu_max_db_size
        self._session = session
        self._subprocess_mode = subprocess_mode
        self._permanently_closed = False
        self.db: Optional[Database] = database
        self.connection: Optional[Connection] = connection

        # Always construct the executor — the shared-lock query path still
        # runs ``blocking_query`` through ``loop.run_in_executor(self.executor,
        # ...)`` and would hit AttributeError without it.
        self.executor = ThreadPoolExecutor()

        if cache_config.shared_ladybug_lock:
            self.redis_lock = get_cache_engine(
                lock_key="ladybug-lock-" + str(uuid5(NAMESPACE_OID, db_path))
            )
        else:
            if subprocess_mode:
                self._ensure_schema()
            else:
                self._initialize_connection()
        self.LADYBUG_ASYNC_LOCK = asyncio.Lock()
        self._connection_lock = asyncio.Lock()
        # Set when ``open_connections == 0``; used by transient teardown
        # paths (e.g. ``delete_graph``) to wait for in-flight queries to
        # finish before dropping native resources. ``close()`` does NOT use
        # this — see ``close()``'s docstring for the cross-loop reason.
        self._all_queries_drained = asyncio.Event()
        self._all_queries_drained.set()
        # Brief sync lock for atomic counter+event mutations. Cannot reuse
        # ``_connection_lock`` here: teardown holds that lock across the
        # ``await`` on ``_all_queries_drained``, and the query's finally
        # needs to decrement+set under SOME lock to make those mutations
        # atomic relative to other queries' increment+clear. If the same
        # lock were reused, the finally would deadlock waiting for
        # teardown to release it. ``threading.Lock`` is held for
        # microseconds (no awaits inside) so it can't deadlock the loop.
        self._counter_lock = threading.Lock()
        # Brief sync lock that makes ``_permanently_closed`` AND the
        # ``self.executor`` reference move together. ``query()`` captures
        # both under this lock; ``close()`` flips closed AND nulls
        # ``self.executor`` under it before shutting the captured
        # executor down. Without this, query could pass the closed check
        # and then call ``run_in_executor`` after close shut the executor
        # down, surfacing "cannot schedule new futures after shutdown".
        # ``threading.Lock`` (not ``asyncio.Lock``) for cross-loop safety:
        # ``close()`` may be invoked from a foreign loop via
        # ``closing_lru_cache._close_value`` running ``asyncio.run``.
        self._lifecycle_lock = threading.Lock()

    def _ensure_schema(self) -> None:
        """Create Node + EDGE tables on the current ``self.connection``.

        Extracted from ``_initialize_connection`` so the subprocess path (where
        the native db/connection are constructed by the factory) can still run
        the same schema bootstrap.
        """
        # Explicit check rather than ``assert`` — assertions are stripped
        # under ``python -O``, which would turn this into a confusing
        # ``AttributeError`` on the next line instead of a clear message.
        if self.connection is None:
            raise RuntimeError("Ladybug connection is not initialized; cannot ensure schema.")
        self.connection.execute("""
            CREATE NODE TABLE IF NOT EXISTS Node(
                id STRING PRIMARY KEY,
                name STRING,
                type STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                properties STRING
            )
        """)
        self.connection.execute("""
            CREATE REL TABLE IF NOT EXISTS EDGE(
                FROM Node TO Node,
                relationship_name STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                properties STRING
            )
        """)
        logger.debug("Ladybug database schema ensured")

    def _initialize_connection(self) -> None:
        """Initialize the Ladybug database connection and schema."""
        # Install the JSON extension via a throwaway DB so its presence is
        # cached before we open the real database. Shared helper lives in
        # cognee_db_workers so the subprocess worker can use the same code
        # without importing cognee. Pass the instance's configured limits
        # so callers that tune ``kuzu_buffer_pool_size`` / ``kuzu_max_db_size``
        # via env or config aren't silently ignored during the install step.
        from cognee_db_workers._kuzu_helpers import install_json_extension_local

        install_json_extension_local(
            buffer_pool_size=self.kuzu_buffer_pool_size,
            max_db_size=self.kuzu_max_db_size,
        )

        try:
            if "s3://" in self.db_path:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    self.temp_graph_file = temp_file.name

                run_sync(self.pull_from_s3())

                self.db = Database(
                    self.temp_graph_file,
                    buffer_pool_size=self.kuzu_buffer_pool_size,
                    max_num_threads=self.kuzu_num_threads,
                    max_db_size=self.kuzu_max_db_size,
                )
            else:
                # Ensure the parent directory exists before creating the database
                db_dir = os.path.dirname(self.db_path)

                # If db_path is just a filename, db_dir will be empty string
                # In this case, use the directory containing the db_path or current directory
                if not db_dir:
                    # If no directory in path, use the absolute path's directory
                    abs_path = os.path.abspath(self.db_path)
                    db_dir = os.path.dirname(abs_path)

                file_storage = get_file_storage(db_dir)

                run_sync(file_storage.ensure_directory_exists())

                try:
                    self.db = Database(
                        self.db_path,
                        buffer_pool_size=self.kuzu_buffer_pool_size,
                        max_num_threads=self.kuzu_num_threads,
                        max_db_size=self.kuzu_max_db_size,
                    )
                except RuntimeError:
                    import ladybug
                    from .ladybug_migrate import needs_migration, ladybug_migration

                    should_migrate, old_version = needs_migration(self.db_path, ladybug.__version__)
                    if should_migrate:
                        ladybug_migration(
                            new_db=self.db_path + "_new",
                            old_db=self.db_path,
                            new_version=ladybug.__version__,
                            old_version=old_version,
                            overwrite=True,
                        )

                    self.db = Database(
                        self.db_path,
                        buffer_pool_size=self.kuzu_buffer_pool_size,
                        max_num_threads=self.kuzu_num_threads,
                        max_db_size=self.kuzu_max_db_size,
                    )

            self.db.init_database()
            self.connection = Connection(self.db)

            try:
                self.connection.execute("LOAD EXTENSION JSON;")
                logger.info("Loaded JSON extension")
            except Exception as e:
                logger.info(f"JSON extension already loaded or unavailable: {e}")

            # Create node table with essential fields and timestamp
            self.connection.execute("""
                CREATE NODE TABLE IF NOT EXISTS Node(
                    id STRING PRIMARY KEY,
                    name STRING,
                    type STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    properties STRING
                )
            """)
            # Create relationship table with timestamp
            self.connection.execute("""
                CREATE REL TABLE IF NOT EXISTS EDGE(
                    FROM Node TO Node,
                    relationship_name STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    properties STRING
                )
            """)
            logger.debug("Ladybug database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ladybug database: {e}")
            raise e

    async def push_to_s3(self) -> None:
        if os.getenv("STORAGE_BACKEND", "").lower() == "s3" and hasattr(self, "temp_graph_file"):
            from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

            s3_file_storage = S3FileStorage("")

            if self.connection:
                async with self.LADYBUG_ASYNC_LOCK:
                    self.connection.execute("CHECKPOINT;")

            s3_file_storage.s3.put(self.temp_graph_file, self.db_path, recursive=True)

    async def pull_from_s3(self) -> None:
        from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

        s3_file_storage = S3FileStorage("")
        try:
            s3_file_storage.s3.get(self.db_path, self.temp_graph_file, recursive=True)
        except FileNotFoundError:
            logger.warning(f"Ladybug S3 storage file not found: {self.db_path}")

    async def is_empty(self) -> bool:
        query = """
        MATCH (n)
        RETURN true
        LIMIT 1;
        """
        query_result = await self.query(query)
        return len(query_result) == 0

    async def query(self, query: str, params: Optional[dict] = None) -> List[Tuple]:
        """
        Execute a Ladybug query asynchronously.

        This method runs a database query while managing lazy connection initialization. It handles
        parameters in a dictionary and processes results to return structured data. The method
        raises any exceptions encountered during query execution.

        Parameters:
        -----------

            - query (str): The Ladybug query string to be executed.
            - params (Optional[dict]): A dictionary of parameters for the query, if applicable.
              (default None)

        Returns:
        --------

            - List[Tuple]: A list of tuples representing the query results.
        """
        # Note on ``close()`` synchronization: actual submission of the
        # blocking work happens under ``_lifecycle_lock`` further down
        # via ``_submit_to_executor_locked``. Capturing the executor
        # reference *before* ``run_in_executor`` would not be enough —
        # ``close()`` could call ``executor.shutdown()`` on the captured
        # ref between capture and submit, surfacing "cannot schedule
        # new futures after shutdown" anyway. Submitting under the lock
        # closes that window: ``close()`` either runs first (and we
        # raise from the helper) or runs after (and ``shutdown(wait=True)``
        # waits for our queued future to complete).
        with new_span("cognee.db.graph.query") as otel_span:
            otel_span.set_attribute(COGNEE_DB_SYSTEM, "ladybug")
            otel_span.set_attribute(COGNEE_DB_QUERY, redact_secrets(query[:500]))

            params = params or {}

            def blocking_query(connection):
                try:
                    result = connection.execute(query, params)
                    rows = []

                    while result.has_next():
                        row = result.get_next()
                        processed_rows = []
                        for val in row:
                            if hasattr(val, "as_py"):
                                val = val.as_py()
                            processed_rows.append(val)
                        rows.append(tuple(processed_rows))

                    return rows
                except Exception as e:
                    logger.error(f"Query execution failed: {str(e)}")
                    raise

            try:
                if cache_config.shared_ladybug_lock:
                    # Shared-lock path: the Redis lock MUST be acquired before
                    # any native ``ladybug.Database`` is opened on this file.
                    # Opening Ladybug takes the on-disk file lock, and if we open
                    # first we race the previous Redis-lock holder that's
                    # still releasing its own native handles.
                    assert self.redis_lock is not None
                    # ``acquire_lock()`` is sync and can block on Redis I/O
                    # for up to ``blocking_timeout`` (default 300 s) waiting
                    # for the previous holder. Offload so we don't freeze
                    # the event loop while another process holds the lock.
                    redis_lock_handle = await asyncio.to_thread(self.redis_lock.acquire_lock)
                    try:
                        # Increment under ``_connection_lock`` so a transient
                        # teardown waiting in ``_drain_in_flight_queries``
                        # can't see ``open_connections == 0`` between the
                        # lock release and our increment. The counter+event
                        # mutation itself uses ``_counter_lock`` so it stays
                        # atomic against other queries' decrement+set.
                        async with self._connection_lock:
                            connection = self.get_or_init_connection()
                            with self._counter_lock:
                                self.open_connections += 1
                                self._all_queries_drained.clear()
                        logger.debug(f"Open connections after open: {self.open_connections}")
                        try:
                            # Submit + check-closed atomically under
                            # ``_lifecycle_lock``. See top-of-method note.
                            future = self._submit_to_executor_locked(blocking_query, connection)
                            result = await asyncio.wrap_future(future)
                        finally:
                            # Decrement under ``_counter_lock`` (not
                            # ``_connection_lock`` — teardown holds the
                            # latter across its ``await`` and we'd deadlock).
                            with self._counter_lock:
                                self.open_connections -= 1
                                if self.open_connections == 0:
                                    self._all_queries_drained.set()
                            logger.debug(f"Open connections after close: {self.open_connections}")
                            # Drop native handles BEFORE releasing the Redis
                            # lock so the next holder can take the on-disk
                            # file lock without fighting us. Drain first for
                            # symmetry with ``delete_graph``: the redis lock
                            # already serializes us in practice, but the
                            # drain costs nothing here (we just decremented
                            # to zero) and stays correct under any future
                            # change to redis-lock reentrancy.
                            async with self._connection_lock:
                                await self._drain_in_flight_queries()
                                if self._subprocess_mode:
                                    await asyncio.to_thread(self._drop_native_resources)
                                else:
                                    self._drop_native_resources()
                    finally:
                        # ``release_lock()`` is also sync and does Redis
                        # I/O — offload for symmetry with the acquire path.
                        await asyncio.to_thread(self.redis_lock.release_lock, redis_lock_handle)
                else:
                    # Hold _connection_lock only for init + counter bookkeeping;
                    # the actual query runs unlocked so multiple queries can
                    # execute concurrently. Counter increment must be inside
                    # ``_connection_lock`` so ``_drain_in_flight_queries``
                    # can't miss us, AND inside ``_counter_lock`` so the
                    # increment+clear is atomic against other queries'
                    # decrement+set in their ``finally``.
                    async with self._connection_lock:
                        connection = self.get_or_init_connection()
                        with self._counter_lock:
                            self.open_connections += 1
                            self._all_queries_drained.clear()
                    try:
                        # Submit + check-closed atomically under
                        # ``_lifecycle_lock``. See top-of-method note.
                        future = self._submit_to_executor_locked(blocking_query, connection)
                        result = await asyncio.wrap_future(future)
                    finally:
                        # Decrement under ``_counter_lock`` (not
                        # ``_connection_lock`` — teardown holds the latter
                        # across its ``await`` and we'd deadlock).
                        with self._counter_lock:
                            self.open_connections -= 1
                            if self.open_connections == 0:
                                self._all_queries_drained.set()

                otel_span.set_attribute(COGNEE_DB_ROW_COUNT, len(result))
                return result
            except Exception as e:
                otel_span.set_status(StatusCode.ERROR, str(e))
                otel_span.record_exception(e)
                raise

    def get_or_init_connection(self) -> Connection:
        """Return the current connection, initializing it first if needed.

        Subprocess mode rebuilds proxies through the surviving
        ``self._session`` rather than falling through to a local
        ``ladybug.Database`` init — opening the same DB path in the main
        process would conflict with the subprocess on the Ladybug file
        lock. If ``self._session`` itself is gone the adapter is a
        permanent error state (only ``close()`` zeroes the session).

        Callers must hold ``_connection_lock`` to prevent races with
        explicit calls to ``close()``.
        """
        # Top-level closed check applies in BOTH modes. Read under
        # ``_lifecycle_lock`` so a concurrent ``close()`` either hasn't
        # started (we proceed) or has already flipped the flag (we
        # raise). ``close()`` latches the flag at the very start of
        # teardown, before touching any resources.
        with self._lifecycle_lock:
            if self._permanently_closed:
                raise RuntimeError("LadybugAdapter is closed; a new adapter must be created.")
        if not self.connection:
            if self._subprocess_mode:
                if self._session is None:
                    raise RuntimeError(
                        "LadybugAdapter subprocess session is gone; adapter "
                        "cannot be re-initialized."
                    )
                self._rebuild_subprocess_proxies()
            else:
                self._initialize_connection()
            # Re-check the closed latch after init: ``close()`` may have
            # flipped it while we were inside ``_initialize_connection``
            # (which opens a ladybug.Database and takes the on-disk file
            # lock). Without this re-check we'd publish the freshly
            # opened native handles onto an already-closed adapter,
            # keeping the file lock alive for the rest of the process.
            with self._lifecycle_lock:
                closed = self._permanently_closed
            if closed:
                self._drop_native_resources()
                raise RuntimeError("LadybugAdapter is closed; a new adapter must be created.")
        # Explicit check rather than ``assert`` — assertions are stripped
        # under ``python -O`` and would degrade to a confusing
        # ``AttributeError`` in callers if init silently failed.
        if self.connection is None:
            raise RuntimeError("LadybugAdapter connection initialization failed.")
        return self.connection

    def _submit_to_executor_locked(self, fn, *args):
        """Atomically check ``_permanently_closed`` AND submit ``fn`` to
        ``self.executor``, all under ``_lifecycle_lock``.

        Submitting under the lock (rather than capturing the executor
        ref and submitting later) is what closes the close-vs-query
        race: if ``close()`` is interleaving, it must take the same
        lock to flip the flag and pull the executor reference. So the
        only two outcomes here are (a) we observe the closed flag and
        raise, or (b) we get a stable reference and ``executor.submit``
        succeeds — at which point ``close()``'s
        ``executor.shutdown(wait=True)`` will wait for our just-queued
        future to complete, instead of refusing to schedule it.

        Returns a ``concurrent.futures.Future``; await
        ``asyncio.wrap_future(future)`` to consume the result on the
        calling event loop.
        """
        with self._lifecycle_lock:
            if self._permanently_closed or self.executor is None:
                raise RuntimeError("LadybugAdapter is closed; a new adapter must be created.")
            return self.executor.submit(fn, *args)

    def _drop_native_resources(self) -> None:
        """Synchronously drop the native Ladybug Database + Connection handles.

        Does **not** latch ``_permanently_closed`` and does **not** touch the
        subprocess session. Used by the shared_ladybug_lock per-query path (where
        we want to release the on-disk file lock between queries) and by
        ``delete_graph`` (which needs the file handles closed before removing
        the db directory). The adapter remains reusable — a subsequent query
        will lazily re-initialize via ``get_or_init_connection``.
        """
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing Ladybug connection: {e}")
            self.connection = None
        if self.db is not None:
            try:
                self.db.close()
            except Exception as e:
                logger.warning(f"Error closing Ladybug database: {e}")
            self.db = None

    def _rebuild_subprocess_proxies(self) -> None:
        """Recreate ``self.db`` + ``self.connection`` against the existing
        ``self._session`` after a transient drop (e.g. files removed by
        ``delete_graph`` or native handles dropped by the shared-lock
        per-query path).

        Subprocess-mode counterpart to local mode's ``_initialize_connection``.
        Called lazily from ``get_or_init_connection`` on the next query, not
        eagerly — that way ``delete_graph`` does not silently recreate the
        on-disk store it just removed. Sync method: the proxy constructors
        issue blocking RPCs through the session, but the call site already
        runs sync from inside ``get_or_init_connection`` (matching the
        local-mode pattern).
        """
        # Imported here to avoid a top-level cycle with the proxy module.
        from cognee.infrastructure.databases.graph.kuzu.subprocess.proxy import (
            RemoteKuzuConnection,
            RemoteKuzuDatabase,
        )

        self.db = RemoteKuzuDatabase(
            self._session,
            db_path=self.db_path,
            buffer_pool_size=self.kuzu_buffer_pool_size,
            max_num_threads=self.kuzu_num_threads,
            max_db_size=self.kuzu_max_db_size,
        )
        self.db.init_database()
        self.connection = RemoteKuzuConnection(self._session, self.db)
        # Re-load the JSON extension on the fresh connection — the
        # original setup path did this and queries that touch JSON would
        # otherwise fail with "extension not loaded" after delete_graph.
        try:
            self.connection.load_extension("JSON")
        except Exception as e:
            logger.warning(f"Could not load JSON extension after reopen: {e}")
        # Recreate the Node/EDGE schema — ``delete_graph`` removed the
        # on-disk store, so the worker is now talking to a fresh empty
        # DB with no tables. Without this, the very next graph query
        # after ``delete_graph`` raises "table Node does not exist".
        self._ensure_schema()

    async def _drain_in_flight_queries(self) -> None:
        """Wait until every query that's currently mid-``run_in_executor``
        has finished. The caller MUST hold ``_connection_lock`` so new
        queries can't start while we wait — otherwise the drain would
        race a fresh increment.

        Used by transient-teardown paths (currently ``delete_graph`` and
        the shared-lock per-query cleanup) so ``_drop_native_resources``
        doesn't tear out a connection an executor thread is still using.
        ``close()`` does NOT use this — see its docstring for why
        (cross-loop ``asyncio.Event.wait()`` would raise).

        Reads the counter under ``_counter_lock`` so a stale ``set()``
        from a finishing query can't race a fresh increment from a new
        one and trick us into busy-spinning on an event that's set while
        ``open_connections > 0``.
        """
        while True:
            with self._counter_lock:
                if self.open_connections == 0:
                    return
            await self._all_queries_drained.wait()

    async def close(self):
        """Permanently close the adapter, releasing native resources and (in
        subprocess mode) shutting down the worker process.

        Intentionally does **not** hold ``_connection_lock``: that lock is an
        ``asyncio.Lock`` bound to the loop on which the adapter was created.
        LRU eviction may invoke ``close()`` from a different loop (for
        example via ``asyncio.run`` in ``closing_lru_cache._close_value``),
        and awaiting a foreign-loop lock raises "got Future attached to a
        different loop". After this call the adapter is not reusable — see
        ``_drop_native_resources`` if you want a transient drop.

        Shuts down our ``ThreadPoolExecutor`` with ``wait=True`` first — this
        serves two purposes: (a) drains any in-flight ``blocking_query``
        submissions, preventing a race where ``close()`` tears down
        ``self.connection`` while an executor thread is still mid-
        ``connection.execute``; (b) reaps the executor threads that would
        otherwise leak on every LRU eviction.

        Note: transient-teardown paths (``delete_graph``) use the asyncio
        ``_drain_in_flight_queries`` helper instead, but that's not safe
        from a foreign loop — ``executor.shutdown(wait=True)`` is the
        cross-loop equivalent and the only correct choice here.

        Idempotent — repeated calls observe ``_permanently_closed`` and
        return early without re-shutting-down anything.
        """
        # Atomically: flip the closed flag, capture the executor
        # reference, and null out ``self.executor``. A concurrent
        # ``query()`` either sees the closed flag (raises clean) or
        # captures a still-live executor (its run_in_executor will
        # complete normally because we shut down with ``wait=True``
        # below). Without nulling self.executor under the lock, a query
        # that captured ``self.executor`` *after* the flag flipped
        # could still submit to the about-to-be-shut-down executor.
        # Idempotent — a second close() sees the flag and returns.
        with self._lifecycle_lock:
            if self._permanently_closed:
                return
            self._permanently_closed = True
            executor = self.executor
            self.executor = None

        # Both ``executor.shutdown(wait=True)`` and
        # ``SubprocessSession.shutdown()`` are sync-blocking calls that can
        # take seconds (executor: thread join; session: join/terminate/kill
        # chain plus a bounded ``_rpc_lock`` acquire). Offload them to a
        # worker thread so awaiting ``close()`` doesn't freeze the calling
        # event loop. ``asyncio.to_thread`` is safe across loops — required
        # because ``close()`` may be invoked from a foreign loop via
        # ``closing_lru_cache._close_value`` running ``asyncio.run``.
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, True)
        # In subprocess mode, ``_drop_native_resources`` calls
        # ``self.connection.close()`` and ``self.db.close()`` which are
        # proxy RPCs through ``session.call(...)`` and can block on
        # the worker for hundreds of ms. Offload to a thread so we
        # don't freeze the event loop. Local mode stays sync — closing
        # an in-process Ladybug Database/Connection is fast.
        if self._subprocess_mode:
            await asyncio.to_thread(self._drop_native_resources)
        else:
            self._drop_native_resources()
        if self._session is not None:
            try:
                await asyncio.to_thread(self._session.shutdown)
            except Exception as e:
                logger.warning(f"Error shutting down Ladybug subprocess: {e}")
            self._session = None
        logger.info("Ladybug database closed successfully")

    @asynccontextmanager
    async def get_session(self):
        """
        Get a database session.

        This provides an API-compatible session management for Ladybug, even though it does not
        have built-in session management like other databases. It yields the current connection
        and on exit performs cleanup if necessary.
        """
        try:
            yield self.connection
        finally:
            pass

    def _parse_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw node result (with JSON properties) into a dictionary."""
        if data.get("properties"):
            try:
                props = json.loads(data["properties"])
                # Remove the JSON field and merge its contents
                data.pop("properties")
                data.update(props)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse properties JSON for node {data.get('id')}")
        return data

    def _parse_node_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if isinstance(data, dict) and "properties" in data and data["properties"]:
                props = json.loads(data["properties"])
                data.update(props)
                del data["properties"]
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse properties JSON for node {data.get('id')}")
            return data

    # Helper method for building edge queries

    def _edge_query_and_params(
        self, from_node: str, to_node: str, relationship_name: str, properties: Dict[str, Any]
    ) -> Tuple[str, dict]:
        """Build the edge creation query and parameters."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        query = """
            MATCH (from:Node), (to:Node)
            WHERE from.id = $from_id AND to.id = $to_id
            MERGE (from)-[r:EDGE {
                relationship_name: $relationship_name
            }]->(to)
            ON CREATE SET
                r.created_at = timestamp($created_at),
                r.updated_at = timestamp($updated_at),
                r.properties = $properties
            ON MATCH SET
                r.updated_at = timestamp($updated_at),
                r.properties = $properties
        """
        params = {
            "from_id": from_node,
            "to_id": to_node,
            "relationship_name": relationship_name,
            "created_at": now,
            "updated_at": now,
            "properties": json.dumps(properties, cls=JSONEncoder),
        }
        return query, params

    # Node Operations

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists.

        This method checks for the existence of a node in the database by its identifier. It
        returns a boolean indicating whether the node is present or not.

        Parameters:
        -----------

            - node_id (str): The identifier of the node to check.

        Returns:
        --------

            - bool: True if the node exists, False otherwise.
        """
        query_str = "MATCH (n:Node) WHERE n.id = $id RETURN COUNT(n) > 0"
        result = await self.query(query_str, {"id": node_id})
        return result[0][0] if result else False

    async def add_node(self, node: DataPoint) -> None:
        """
        Add a single node to the graph if it doesn't exist.

        This method constructs and executes a query to add a node to the graph, ensuring that it
        is not duplicated by checking its existence first. An error is raised if the operation
        fails.

        Parameters:
        -----------

            - node (DataPoint): The node to be added, represented as a DataPoint.
        """
        try:
            properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

            # Extract core fields with defaults if not present
            core_properties = {
                "id": str(properties.get("id", "")),
                "name": str(properties.get("name", "")),
                "type": str(properties.get("type", "")),
            }

            # Remove core fields from other properties
            for key in core_properties:
                properties.pop(key, None)

            core_properties["properties"] = json.dumps(properties, cls=JSONEncoder)

            # Add timestamps for new node
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
            fields = []
            params = {}
            for key, value in core_properties.items():
                if value is not None:
                    param_name = f"param_{key}"
                    fields.append(f"{key}: ${param_name}")
                    params[param_name] = value

            # Add timestamp fields
            fields.extend(
                ["created_at: timestamp($created_at)", "updated_at: timestamp($updated_at)"]
            )
            params.update({"created_at": now, "updated_at": now})

            merge_query = f"""
            MERGE (n:Node {{id: $param_id}})
            ON CREATE SET n += {{{", ".join(fields)}}}
            """
            await self.query(merge_query, params)

        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    async def add_nodes(self, nodes: List[DataPoint]) -> None:
        """
        Add multiple nodes to the graph in a batch operation.

        This method allows for the addition of multiple nodes in a single operation to enhance
        performance. It processes a list of nodes and constructs the necessary query for
        insertion. Errors encountered during the addition will be logged and raised.

        Parameters:
        -----------

            - nodes (List[DataPoint]): A list of nodes to be added to the graph, each
              represented as a DataPoint.
        """
        if not nodes:
            return

        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

            # Prepare all nodes data
            node_params = []
            for node in nodes:
                properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

                core_properties = {
                    "id": str(properties.get("id", "")),
                    "name": str(properties.get("name", "")),
                    "type": str(properties.get("type", "")),
                }

                # Remove core fields from other properties
                for key in core_properties:
                    properties.pop(key, None)

                node_params.append(
                    {
                        **core_properties,
                        "properties": json.dumps(properties, cls=JSONEncoder),
                        "created_at": now,
                        "updated_at": now,
                    }
                )

            if node_params:
                # Batch merge nodes
                merge_query = """
                UNWIND $nodes AS node
                MERGE (n:Node {id: node.id})
                ON CREATE SET
                    n.name = node.name,
                    n.type = node.type,
                    n.properties = node.properties,
                    n.created_at = timestamp(node.created_at),
                    n.updated_at = timestamp(node.updated_at)
                ON MATCH SET
                    n.name = node.name,
                    n.type = node.type,
                    n.properties = node.properties,
                    n.updated_at = timestamp(node.updated_at)
                """
                await self.query(merge_query, {"nodes": node_params})
                logger.debug(f"Processed {len(node_params)} nodes in batch")

        except Exception as e:
            logger.error(f"Failed to add nodes in batch: {e}")
            raise

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node and its relationships.

        This method removes a node identified by its ID along with all associated relationships.
        It encapsulates the delete operation for simplicity in usage.

        Parameters:
        -----------

            - node_id (str): The identifier of the node to be deleted.
        """
        query_str = "MATCH (n:Node) WHERE n.id = $id DETACH DELETE n"
        await self.query(query_str, {"id": node_id})

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete multiple nodes at once.

        This method facilitates the deletion of a list of nodes, identified by their IDs,
        concurrently. It ensures efficiency by using a single query to detach deletes for all
        nodes in the list.

        Parameters:
        -----------

            - node_ids (List[str]): A list of identifiers for the nodes to be deleted.
        """
        query_str = "MATCH (n:Node) WHERE n.id IN $ids DETACH DELETE n"
        await self.query(query_str, {"ids": node_ids})

    async def extract_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract a node by its ID.

        This method retrieves a node's data by its identifier and returns it as a dictionary. If
        the node is not found or an error occurs, it returns None.

        Parameters:
        -----------

            - node_id (str): The identifier of the node to be extracted.

        Returns:
        --------

            - Optional[Dict[str, Any]]: A dictionary of the node's properties if found,
              otherwise None.
        """
        query_str = """
        MATCH (n:Node)
        WHERE n.id = $id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            if result and result[0]:
                node_data = self._parse_node(result[0][0])
                return node_data
            return None
        except Exception as e:
            logger.error(f"Failed to extract node {node_id}: {e}")
            return None

    async def extract_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Extract multiple nodes by their IDs.

        This method retrieves a list of nodes identified by their IDs and returns their data as
        a list of dictionaries. It handles possible retrieval errors internally and will return
        an empty list if no nodes are found.

        Parameters:
        -----------

            - node_ids (List[str]): A list of identifiers for the nodes to be extracted.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries containing the properties of the
              extracted nodes.
        """
        query_str = """
        MATCH (n:Node)
        WHERE n.id IN $node_ids
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_ids": node_ids})
            # Parse each node using the same helper function
            nodes = [self._parse_node(row[0]) for row in results if row[0]]
            return nodes
        except Exception as e:
            logger.error(f"Failed to extract nodes: {e}")
            return []

    # Edge Operations

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        """
        Check if an edge exists between nodes with the given relationship name.

        This method verifies the existence of a directed edge defined by the relationship name
        between two specified nodes. It returns a boolean value indicating presence or absence
        of the edge.

        Parameters:
        -----------

            - from_node (str): The identifier of the source node.
            - to_node (str): The identifier of the target node.
            - edge_label (str): The label of the edge representing the relationship name.

        Returns:
        --------

            - bool: True if the edge exists, False otherwise.
        """
        query_str = """
        MATCH (from:Node)-[r:EDGE]->(to:Node)
        WHERE from.id = $from_id AND to.id = $to_id AND r.relationship_name = $edge_label
        RETURN COUNT(r) > 0
        """
        result = await self.query(
            query_str, {"from_id": from_node, "to_id": to_node, "edge_label": edge_label}
        )
        return result[0][0] if result else False

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Check if multiple edges exist in a batch operation.

        This method checks for the presence of specified edges in the database and returns a
        list of edges that exist. It is beneficial for efficiency in checking multiple edges
        simultaneously.

        Parameters:
        -----------

            - edges (List[Tuple[str, str, str]]): A list of edges where each edge is represented
              as a tuple of (from_node, to_node, edge_label).

        Returns:
        --------

            - List[Tuple[str, str, str]]: A list of tuples representing the existing edges from
              the provided list.
        """
        if not edges:
            return []

        try:
            # Transform edges into format needed for batch query
            edge_params = [
                {
                    "from_id": str(from_node),  # Ensure string type
                    "to_id": str(to_node),  # Ensure string type
                    "relationship_name": str(edge_label),  # Ensure string type
                }
                for from_node, to_node, edge_label in edges
            ]

            # Batch check query with direct string comparison
            query = """
            UNWIND $edges AS edge
            MATCH (from:Node)-[r:EDGE]->(to:Node)
            WHERE from.id = edge.from_id
            AND to.id = edge.to_id
            AND r.relationship_name = edge.relationship_name
            RETURN from.id, to.id, r.relationship_name
            """

            results = await self.query(query, {"edges": edge_params})

            # Convert results back to tuples and ensure string types
            existing_edges = [(str(row[0]), str(row[1]), str(row[2])) for row in results]

            logger.debug(f"Found {len(existing_edges)} existing edges out of {len(edges)} checked")
            return existing_edges

        except Exception as e:
            logger.error(f"Failed to check edges in batch: {e}")
            return []

    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = {},
    ) -> None:
        """
        Add an edge between two nodes.

        This method constructs and executes a query to create a directed edge between two
        specified nodes with certain properties. It will raise an error if the addition fails
        during execution.

        Parameters:
        -----------

            - from_node (str): The identifier of the source node from which the edge originates.
            - to_node (str): The identifier of the target node to which the edge points.
            - relationship_name (str): The label of the edge to be created, representing the
              relationship name.
            - edge_properties (Dict[str, Any]): A dictionary containing properties for the edge.
              (default {})
        """
        try:
            query, params = self._edge_query_and_params(
                from_node, to_node, relationship_name, edge_properties
            )
            await self.query(query, params)
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            raise

    async def add_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        """
        Add multiple edges in a batch operation.

        This method enables efficient insertion of multiple edges at once by processing a list
        of edge details. It improves performance for batch operations compared to adding edges
        individually. Errors during execution are logged and raised as necessary.

        Parameters:
        -----------

            - edges (List[Tuple[str, str, str, Dict[str, Any]]]): A list of edges represented as
              tuples of (from_node, to_node, relationship_name, edge_properties).
        """
        if not edges:
            return

        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

            edge_params = [
                {
                    "from_id": from_node,
                    "to_id": to_node,
                    "relationship_name": relationship_name,
                    "properties": json.dumps(properties, cls=JSONEncoder),
                    "created_at": now,
                    "updated_at": now,
                }
                for from_node, to_node, relationship_name, properties in edges
            ]

            query = """
            UNWIND $edges AS edge
            MATCH (from:Node), (to:Node)
            WHERE from.id = edge.from_id AND to.id = edge.to_id
            MERGE (from)-[r:EDGE {
                relationship_name: edge.relationship_name
            }]->(to)
            ON CREATE SET
                r.created_at = timestamp(edge.created_at),
                r.updated_at = timestamp(edge.updated_at),
                r.properties = edge.properties
            ON MATCH SET
                r.updated_at = timestamp(edge.updated_at),
                r.properties = edge.properties
            """

            await self.query(query, {"edges": edge_params})

        except Exception as e:
            logger.error(f"Failed to add edges in batch: {e}")
            raise

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """
        Get all edges connected to a node.

        This method retrieves all edges that are linked to a specified node and returns them in
        a structured format. If an error occurs or no edges exist, an empty list is returned.

        Parameters:
        -----------

            - node_id (str): The identifier of the node for which to retrieve edges.

        Returns:
        --------

            - List[Tuple[Dict[str, Any], str, Dict[str, Any]]]: A list of tuples where each
              tuple contains (source_node, relationship_name, target_node), with source_node and
              target_node as dictionaries of node properties.
        """
        query_str = """
        MATCH (n:Node)-[r]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        },
        r.relationship_name,
        {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    source_node = self._parse_node_properties(row[0])
                    target_node = self._parse_node_properties(row[2])
                    edges.append((source_node, row[1], target_node))
            return edges
        except Exception as e:
            logger.error(f"Failed to get edges for node {node_id}: {e}")
            return []

    # Neighbor Operations

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all neighboring nodes.

        This method simply calls the get_neighbours method for API compatibility and retrieves
        connected nodes neighboring the specified node. It returns a list of neighbor nodes'
        properties as dictionaries.

        Parameters:
        -----------

            - node_id (str): The identifier of the node for which to find neighbors.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries representing neighboring nodes'
              properties.
        """
        query_str = """
        MATCH (n:Node)-[r]-(m:Node)
        WHERE n.id = $id
        RETURN DISTINCT {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            return [self._parse_node_properties(row[0]) for row in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get neighbours for node {node_id}: {e}")
            return []

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single node by ID.

        This method retrieves the properties of a node identified by its ID and returns them as
        a dictionary. If the node does not exist, None is returned.

        Parameters:
        -----------

            - node_id (str): The identifier of the node to retrieve.

        Returns:
        --------

            - Optional[Dict[str, Any]]: A dictionary containing the properties of the node if
              found, otherwise None.
        """
        query_str = """
        MATCH (n:Node)
        WHERE n.id = $id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            if result and result[0]:
                return self._parse_node(result[0][0])
            return None
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple nodes by their IDs.

        This method retrieves properties for multiple nodes identified by their IDs and returns
        them as a list of dictionaries. An empty list is returned if no nodes are found or an
        error occurs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of identifiers for the nodes to be retrieved.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries containing properties of each
              retrieved node.
        """
        query_str = """
        MATCH (n:Node)
        WHERE n.id IN $node_ids
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_ids": node_ids})
            return [self._parse_node(row[0]) for row in results if row[0]]
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []

    def _rows_to_dicts(self, rows: List, column_names: List[str]) -> List[Dict[str, Any]]:
        """Convert query result rows to a list of dicts keyed by column names."""
        result = []
        for row in rows:
            if not row or len(row) < len(column_names):
                continue
            result.append(dict(zip(column_names, row)))
        return result

    @staticmethod
    def _resolve_edge_object_id(
        properties: Dict[str, Any], edge_object_id_json: Optional[str]
    ) -> Optional[str]:
        """Resolve edge_object_id from properties or from edge_object_id_json string."""
        edge_object_id = properties.get("edge_object_id")
        if (not isinstance(edge_object_id, str) or not edge_object_id) and isinstance(
            edge_object_id_json, str
        ):
            try:
                parsed = json.loads(edge_object_id_json)
                edge_object_id = parsed if isinstance(parsed, str) else None
            except (TypeError, json.JSONDecodeError):
                edge_object_id = None
        return edge_object_id if isinstance(edge_object_id, str) and edge_object_id else None

    _EDGE_BY_OBJECT_ID_COLUMNS = [
        "from_id",
        "to_id",
        "relationship_name",
        "edge_object_id_json",
        "properties",
    ]

    async def _fetch_edge_rows_by_object_ids(
        self, edge_object_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """Fetch edge rows (as dicts) for the given edge_object_ids."""
        if not edge_object_ids:
            return []
        requested_ids_json = [json.dumps(eid) for eid in edge_object_ids]
        query = """
        MATCH (from:Node)-[r:EDGE]->(to:Node)
        WITH from, to, r, CAST(json_extract(r.properties, '$.edge_object_id') AS STRING) AS edge_object_id_json
        WHERE edge_object_id_json IN $edge_object_ids_json
        RETURN from.id AS from_id, to.id AS to_id, r.relationship_name AS relationship_name,
               edge_object_id_json AS edge_object_id_json, r.properties AS properties
        """
        rows = await self.query(query, {"edge_object_ids_json": requested_ids_json})
        return self._rows_to_dicts(rows, self._EDGE_BY_OBJECT_ID_COLUMNS)

    def _build_node_feedback_updates(
        self,
        nodes: List[Dict[str, Any]],
        node_feedback_weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build UNWIND items for node feedback weight updates."""
        updates = []
        for node in nodes:
            node_id = node.get("id")
            if not isinstance(node_id, str) or node_id not in node_feedback_weights:
                continue
            properties = {
                k: v
                for k, v in node.items()
                if k not in {"id", "name", "type", "created_at", "updated_at"}
            }
            properties["feedback_weight"] = float(node_feedback_weights[node_id])
            updates.append(
                {"node_id": node_id, "properties": json.dumps(properties, cls=JSONEncoder)}
            )
        return updates

    async def _execute_node_feedback_updates(self, updates: List[Dict[str, Any]]) -> Set[str]:
        """Run node feedback weight UNWIND/SET; return set of updated node_ids."""
        if not updates:
            return set()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        query = """
        UNWIND $items AS item
        MATCH (n:Node)
        WHERE n.id = item.node_id
        SET n.properties = item.properties,
            n.updated_at = timestamp($updated_at)
        RETURN n.id AS node_id
        """
        result = await self.query(query, {"items": updates, "updated_at": now})
        rows_dicts = self._rows_to_dicts(result, ["node_id"])
        return {str(r["node_id"]) for r in rows_dicts if r.get("node_id") is not None}

    def _build_edge_feedback_updates(
        self,
        edge_rows: List[Dict[str, Any]],
        edge_feedback_weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build UNWIND items for edge feedback weight updates."""
        edge_updates = []
        for row in edge_rows:
            properties_raw = row.get("properties")
            if not properties_raw:
                continue
            try:
                properties = json.loads(properties_raw)
            except (TypeError, json.JSONDecodeError):
                continue
            edge_object_id = self._resolve_edge_object_id(
                properties, row.get("edge_object_id_json")
            )
            if not edge_object_id or edge_object_id not in edge_feedback_weights:
                continue
            properties["feedback_weight"] = float(edge_feedback_weights[edge_object_id])
            edge_updates.append(
                {
                    "edge_object_id": edge_object_id,
                    "from_id": str(row.get("from_id")),
                    "to_id": str(row.get("to_id")),
                    "relationship_name": str(row.get("relationship_name")),
                    "properties": json.dumps(properties, cls=JSONEncoder),
                }
            )
        return edge_updates

    async def _execute_edge_feedback_updates(self, edge_updates: List[Dict[str, Any]]) -> Set[str]:
        """Run edge feedback weight UNWIND/SET; return set of updated edge_object_ids."""
        if not edge_updates:
            return set()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        query = """
        UNWIND $items AS item
        MATCH (from:Node)-[r:EDGE]->(to:Node)
        WHERE from.id = item.from_id
          AND to.id = item.to_id
          AND r.relationship_name = item.relationship_name
        SET r.properties = item.properties,
            r.updated_at = timestamp($updated_at)
        RETURN item.edge_object_id AS edge_object_id
        """
        result = await self.query(query, {"items": edge_updates, "updated_at": now})
        rows_dicts = self._rows_to_dicts(result, ["edge_object_id"])
        return {str(r["edge_object_id"]) for r in rows_dicts if r.get("edge_object_id") is not None}

    async def get_node_feedback_weights(self, node_ids: List[str]) -> Dict[str, float]:
        if not node_ids:
            return {}
        valid_node_ids = [node_id for node_id in node_ids if isinstance(node_id, str) and node_id]
        if not valid_node_ids:
            return {}
        nodes = await self.get_nodes(valid_node_ids)
        result: Dict[str, float] = {}
        for node in nodes:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            value = node.get("feedback_weight", 0.5)
            try:
                result[node_id] = float(value)
            except (TypeError, ValueError):
                result[node_id] = 0.5
        return result

    async def set_node_feedback_weights(
        self, node_feedback_weights: Dict[str, float]
    ) -> Dict[str, bool]:
        if not node_feedback_weights:
            return {}
        node_ids = list(node_feedback_weights.keys())
        valid_node_ids = [nid for nid in node_ids if isinstance(nid, str) and nid]
        if not valid_node_ids:
            return {nid: False for nid in node_ids}
        nodes = await self.get_nodes(valid_node_ids)
        updates = self._build_node_feedback_updates(nodes, node_feedback_weights)
        if not updates:
            return {nid: False for nid in node_ids}
        updated_ids = await self._execute_node_feedback_updates(updates)
        return {nid: (nid in updated_ids) for nid in node_ids}

    async def get_edge_feedback_weights(self, edge_object_ids: List[str]) -> Dict[str, float]:
        if not edge_object_ids:
            return {}
        requested_ids = {eid for eid in edge_object_ids if isinstance(eid, str) and eid}
        if not requested_ids:
            return {}
        edge_rows = await self._fetch_edge_rows_by_object_ids(requested_ids)
        result: Dict[str, float] = {}
        for row in edge_rows:
            properties_raw = row.get("properties")
            if not properties_raw:
                continue
            try:
                properties = json.loads(properties_raw)
            except (TypeError, json.JSONDecodeError):
                continue
            edge_object_id = self._resolve_edge_object_id(
                properties, row.get("edge_object_id_json")
            )
            if not edge_object_id or edge_object_id not in requested_ids:
                continue
            value = properties.get("feedback_weight", 0.5)
            try:
                result[edge_object_id] = float(value)
            except (TypeError, ValueError):
                result[edge_object_id] = 0.5
        return result

    async def set_edge_feedback_weights(
        self, edge_feedback_weights: Dict[str, float]
    ) -> Dict[str, bool]:
        if not edge_feedback_weights:
            return {}
        requested_ids = {eid for eid in edge_feedback_weights if isinstance(eid, str) and eid}
        if not requested_ids:
            return {eid: False for eid in edge_feedback_weights}
        edge_rows = await self._fetch_edge_rows_by_object_ids(requested_ids)
        edge_updates = self._build_edge_feedback_updates(edge_rows, edge_feedback_weights)
        if not edge_updates:
            return {eid: False for eid in edge_feedback_weights}
        updated_ids = await self._execute_edge_feedback_updates(edge_updates)
        return {eid: (eid in updated_ids) for eid in edge_feedback_weights}

    async def get_predecessors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all predecessor nodes.

        This method retrieves all nodes that are predecessors of the specified node. If an edge
        label is provided, it filters the results accordingly. It returns a list of dictionaries
        containing properties of these predecessor nodes.

        Parameters:
        -----------

            - node_id (Union[str, UUID]): The identifier of the specified node.
            - edge_label (Optional[str]): An optional label to filter the edges by relationship
              name. (default None)

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries representing all predecessor nodes'
              properties.
        """
        try:
            if edge_label:
                query_str = """
                MATCH (n)<-[r:EDGE]-(m)
                WHERE n.id = $id AND r.relationship_name = $edge_label
                RETURN properties(m)
                """
                params = {"id": str(node_id), "edge_label": edge_label}
            else:
                query_str = """
                MATCH (n)<-[r:EDGE]-(m)
                WHERE n.id = $id
                RETURN properties(m)
                """
                params = {"id": str(node_id)}
            result = await self.query(query_str, params)
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get predecessors for node {node_id}: {e}")
            return []

    async def get_successors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all successor nodes.

        This method retrieves all nodes that are successors of the specified node. An edge label
        can be provided to filter the results. It returns a list of dictionaries detailing these
        successor nodes' properties.

        Parameters:
        -----------

            - node_id (Union[str, UUID]): The identifier of the specified node.
            - edge_label (Optional[str]): An optional label to filter the edges by relationship
              name. (default None)

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries representing all successor nodes'
              properties.
        """
        try:
            if edge_label:
                query_str = """
                MATCH (n)-[r:EDGE]->(m)
                WHERE n.id = $id AND r.relationship_name = $edge_label
                RETURN properties(m)
                """
                params = {"id": str(node_id), "edge_label": edge_label}
            else:
                query_str = """
                MATCH (n)-[r:EDGE]->(m)
                WHERE n.id = $id
                RETURN properties(m)
                """
                params = {"id": str(node_id)}
            result = await self.query(query_str, params)
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get successors for node {node_id}: {e}")
            return []

    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """
        Get all nodes connected to a given node.

        This method retrieves all nodes directly connected to a specified node along with the
        relationships between them, returning structured data in a list of tuples. Each tuple
        contains source and target node properties along with the relationship information.

        Parameters:
        -----------

            - node_id (str): The identifier of the node for which to retrieve connections.

        Returns:
        --------

            - List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]}: A list of tuples
              containing (source_node, relationship_name, target_node) with dictionaries for
              source_node and target_node properties.
        """
        query_str = """
        MATCH (n:Node)-[r:EDGE]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        },
        {
            relationship_name: r.relationship_name,
            properties: r.properties
        },
        {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    processed_rows = []
                    for i, item in enumerate(row):
                        if isinstance(item, dict):
                            if "properties" in item and item["properties"]:
                                try:
                                    props = json.loads(item["properties"])
                                    item.update(props)
                                    del item["properties"]
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON properties for node/edge {i}"
                                    )
                        processed_rows.append(item)
                    edges.append(tuple(processed_rows))
            return edges if edges else []  # Always return a list, even if empty
        except Exception as e:
            logger.error(f"Failed to get connections for node {node_id}: {e}")
            return []  # Return empty list on error

    async def remove_connection_to_predecessors_of(
        self, node_ids: List[str], edge_label: str
    ) -> None:
        """
        Remove all incoming edges of specified type for given nodes.

        This method disconnects predecessor relationships of a specific type for the specified
        nodes, managing edges in a single operation effectively.

        Parameters:
        -----------

            - node_ids (List[str]): A list of identifiers for the nodes whose relationships to
              be removed.
            - edge_label (str): The label of the edge to be removed.
        """
        query_str = """
        MATCH (n)<-[r:EDGE]-(m)
        WHERE n.id IN $node_ids AND r.relationship_name = $edge_label
        DELETE r
        """
        await self.query(query_str, {"node_ids": node_ids, "edge_label": edge_label})

    async def remove_connection_to_successors_of(
        self, node_ids: List[str], edge_label: str
    ) -> None:
        """
        Remove all outgoing edges of specified type for given nodes.

        This method disconnects successor relationships of a specified type for the specified
        nodes in a single efficient operation.

        Parameters:
        -----------

            - node_ids (List[str]): A list of identifiers for the nodes whose relationships to
              be removed.
            - edge_label (str): The label of the edge to be removed.
        """
        query_str = """
        MATCH (n)-[r:EDGE]->(m)
        WHERE n.id IN $node_ids AND r.relationship_name = $edge_label
        DELETE r
        """
        await self.query(query_str, {"node_ids": node_ids, "edge_label": edge_label})

    # Graph-wide Operations

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """
        Get all nodes and edges in the graph.

        This method fetches the entire graph's structure, including all nodes and their
        properties as well as relationships and their details, returning them in a structured
        format. Errors during query execution will result in raised exceptions.

        Returns:
        --------

            - Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
              A tuple with two elements: a list of tuples of (node_id, properties) and a list of
              tuples of (source_id, target_id, relationship_name, properties).
        """

        import time

        start_time = time.time()

        try:
            nodes_query = """
            MATCH (n:Node)
            RETURN n.id, {
                name: n.name,
                type: n.type,
                properties: n.properties
            }
            """
            nodes = await self.query(nodes_query)
            formatted_nodes = []
            for n in nodes:
                if n[0]:
                    node_id = str(n[0])
                    props = n[1]
                    if props.get("properties"):
                        try:
                            additional_props = json.loads(props["properties"])
                            props.update(additional_props)
                            del props["properties"]
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse properties JSON for node {node_id}")
                    formatted_nodes.append((node_id, props))
            if not formatted_nodes:
                logger.warning("No nodes found in the database")
                return [], []

            edges_query = """
            MATCH (n:Node)-[r]->(m:Node)
            RETURN n.id, m.id, r.relationship_name, r.properties
            """
            edges = await self.query(edges_query)
            formatted_edges = []
            for e in edges:
                if e and len(e) >= 3:
                    source_id = str(e[0])
                    target_id = str(e[1])
                    rel_type = str(e[2])
                    props = {}
                    if len(e) > 3 and e[3]:
                        try:
                            props = json.loads(e[3])
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"Failed to parse edge properties for {source_id}->{target_id}"
                            )
                    formatted_edges.append((source_id, target_id, rel_type, props))

            if formatted_nodes and not formatted_edges:
                logger.debug("No edges found, creating self-referential edges for nodes")
                for node_id, _ in formatted_nodes:
                    formatted_edges.append(
                        (
                            node_id,
                            node_id,
                            "SELF",
                            {
                                "relationship_name": "SELF",
                                "relationship_type": "SELF",
                                "vector_distance": 0.0,
                            },
                        )
                    )

            retrieval_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(nodes)} nodes and {len(edges)} edges in {retrieval_time:.2f} seconds"
            )
            return formatted_nodes, formatted_edges
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            raise

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """
        Get the k-hop neighborhood subgraph around a set of seed nodes.

        Returns all nodes and edges within `depth` hops of any seed node,
        in the same format as get_graph_data().
        """
        import time

        start_time = time.time()

        try:
            if not node_ids:
                logger.warning("No node IDs provided for neighborhood retrieval.")
                return [], []

            # Use variable-length path to find all nodes within depth hops
            path_query = f"""
            MATCH (seed:Node)-[r*1..{depth}]-(neighbor:Node)
            WHERE seed.id IN $node_ids{" AND ALL(rel IN r WHERE rel.relationship_name IN $edge_types)" if edge_types else ""}
            RETURN DISTINCT neighbor.id
            """
            params = {"node_ids": node_ids}
            if edge_types:
                params["edge_types"] = edge_types

            neighbor_rows = await self.query(path_query, params)
            neighbor_ids = [row[0] for row in neighbor_rows if row[0]]

            # Combine seed nodes and neighbor nodes
            all_ids = list(set(node_ids) | set(neighbor_ids))

            # Fetch all nodes
            nodes_query = """
            MATCH (n:Node)
            WHERE n.id IN $ids
            RETURN n.id, {
                name: n.name,
                type: n.type,
                properties: n.properties
            }
            """
            node_rows = await self.query(nodes_query, {"ids": all_ids})
            formatted_nodes = []
            for n in node_rows:
                if n[0]:
                    node_id = str(n[0])
                    props = n[1]
                    if props.get("properties"):
                        try:
                            additional_props = json.loads(props["properties"])
                            props.update(additional_props)
                            del props["properties"]
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse properties JSON for node {node_id}")
                    formatted_nodes.append((node_id, props))

            if not formatted_nodes:
                logger.warning("No nodes found in neighborhood.")
                return [], []

            # Fetch all edges between the collected nodes
            edges_query = """
            MATCH (n:Node)-[r]->(m:Node)
            WHERE n.id IN $ids AND m.id IN $ids
            RETURN n.id, m.id, r.relationship_name, r.properties
            """
            edge_rows = await self.query(edges_query, {"ids": all_ids})
            formatted_edges = []
            for e in edge_rows:
                if e and len(e) >= 3:
                    source_id = str(e[0])
                    target_id = str(e[1])
                    rel_type = str(e[2])
                    props = {}
                    if len(e) > 3 and e[3]:
                        try:
                            props = json.loads(e[3])
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"Failed to parse edge properties for {source_id}->{target_id}"
                            )
                    formatted_edges.append((source_id, target_id, rel_type, props))

            retrieval_time = time.time() - start_time
            logger.info(
                f"Neighborhood retrieval ({depth}-hop): {len(formatted_nodes)} nodes and "
                f"{len(formatted_edges)} edges in {retrieval_time:.2f}s"
            )
            return formatted_nodes, formatted_edges

        except Exception as e:
            logger.error(f"Failed to get neighborhood: {e}")
            raise

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        """
        Get subgraph for a set of nodes based on type and names.

        This method queries for nodes of a specific type and their corresponding neighbors,
        returning both nodes and edges connecting them. It's useful for analyzing a targeted
        subset of the graph.

        Parameters:
        -----------

            - node_type (Type[Any]): Type of nodes to retrieve as specified by the user.
            - node_name (List[str]): List of names corresponding to the nodes to be retrieved.

        Returns:
        --------

            - Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]}: A tuple
              containing a list of nodes and a list of edges related to those nodes.
        """
        label = node_type.__name__
        primary_query = """
            UNWIND $names AS wantedName
            MATCH (n:Node)
            WHERE n.type = $label AND n.name = wantedName
            RETURN DISTINCT n.id
        """
        primary_rows = await self.query(primary_query, {"names": node_name, "label": label})
        primary_ids = [row[0] for row in primary_rows]
        if not primary_ids:
            return [], []

        if node_name_filter_operator == "OR":
            neighbor_query = """
                MATCH (n:Node)-[:EDGE]-(nbr:Node)
                WHERE n.id IN $ids
                RETURN DISTINCT nbr.id
            """
            params = {"ids": primary_ids}
        else:
            neighbor_query = """
                MATCH (n:Node)-[:EDGE]-(nbr:Node)
                WHERE n.id IN $ids
                WITH nbr.id AS nbr_id, COUNT(DISTINCT n.id) AS matched_count
                WHERE matched_count = $primary_count
                RETURN nbr_id
            """
            params = {"ids": primary_ids, "primary_count": len(primary_ids)}

        nbr_rows = await self.query(neighbor_query, params)
        neighbor_ids = [row[0] for row in nbr_rows]

        all_ids = list({*primary_ids, *neighbor_ids})

        nodes_query = """
            MATCH (n:Node)
            WHERE n.id IN $ids
            RETURN n.id, n.name, n.type, n.properties
        """
        node_rows = await self.query(nodes_query, {"ids": all_ids})
        nodes: List[Tuple[str, dict]] = []
        for node_id, name, typ, props in node_rows:
            data = {"id": node_id, "name": name, "type": typ}
            if props:
                try:
                    data.update(json.loads(props))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON props for node {node_id}")
            nodes.append((node_id, data))

        edges_query = """
            MATCH (a:Node)-[r:EDGE]-(b:Node)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id, b.id, r.relationship_name, r.properties
        """
        edge_rows = await self.query(edges_query, {"ids": all_ids})
        edges: List[Tuple[str, str, str, dict]] = []
        for from_id, to_id, rel_type, props in edge_rows:
            data = {}
            if props:
                try:
                    data = json.loads(props)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON props for edge {from_id}->{to_id}")

            edges.append((from_id, to_id, rel_type, data))

        return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ):
        """
        Get filtered nodes and relationships based on attributes.

        This method accepts attribute filters and retrieves nodes and relationships that match
        the specified conditions. It allows complex filtering across node properties and edge
        attributes.

        Parameters:
        -----------

            - attribute_filters (List[Dict[str, List[Union[str, int]]]]): A list of dictionaries
              specifying attributes and their corresponding values for filtering nodes and
              edges.

        Returns:
        --------

            A tuple containing a list of filtered node properties and a list of filtered edge
            properties.
        """
        where_clauses = []
        params = {}

        if not attribute_filters:
            return [], []

        for i, filter_dict in enumerate(attribute_filters):
            for attr, values in filter_dict.items():
                if not attr.isidentifier():
                    raise CogneeValidationError(
                        f"Invalid attribute filter key '{attr}'. Only identifiers are allowed."
                    )
                if not values:
                    continue

                param_name = f"values_{i}_{attr}"
                where_clauses.append(f"n.{attr} IN ${param_name}")
                params[param_name] = values

        if not where_clauses:
            return [], []

        where_clause = " AND ".join(where_clauses)
        nodes_query = f"""
        MATCH (n:Node)
        WHERE {where_clause}
        RETURN n.id, {{
            name: n.name,
            type: n.type,
            properties: n.properties
        }}
        """
        edges_query = f"""
        MATCH (n1:Node)-[r:EDGE]->(n2:Node)
        WHERE {where_clause.replace("n.", "n1.")} AND {where_clause.replace("n.", "n2.")}
        RETURN n1.id, n2.id, r.relationship_name, r.properties
        """
        nodes, edges = await asyncio.gather(
            self.query(nodes_query, params), self.query(edges_query, params)
        )
        formatted_nodes = []
        for n in nodes:
            if n[0]:
                node_id = str(n[0])
                props = n[1]
                if props.get("properties"):
                    try:
                        additional_props = json.loads(props["properties"])
                        props.update(additional_props)
                        del props["properties"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse properties JSON for node {node_id}")
                formatted_nodes.append((node_id, props))
        if not formatted_nodes:
            logger.warning("No nodes found in the database")
            return [], []

        formatted_edges = []
        for e in edges:
            if e and len(e) >= 3:
                source_id = str(e[0])
                target_id = str(e[1])
                rel_type = str(e[2])
                props = {}
                if len(e) > 3 and e[3]:
                    try:
                        props = json.loads(e[3])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            f"Failed to parse edge properties for {source_id}->{target_id}"
                        )
                formatted_edges.append((source_id, target_id, rel_type, props))
        return formatted_nodes, formatted_edges

    async def get_id_filtered_graph_data(self, target_ids: list[str]):
        """
        Retrieve graph data filtered by specific node IDs, including their direct neighbors
        and only edges where one endpoint matches those IDs.

        Returns:
            nodes: List[dict]   -> Each dict includes "id" and all node properties
            edges: List[dict]   -> Each dict includes "source", "target", "type", "properties"
        """
        import time

        start_time = time.time()

        try:
            if not target_ids:
                logger.warning("No target IDs provided for ID-filtered graph retrieval.")
                return [], []

            if not all(isinstance(x, str) for x in target_ids):
                raise CogneeValidationError("target_ids must be a list of strings")

            query = """
            MATCH (n:Node)-[r]->(m:Node)
            WHERE n.id IN $target_ids OR m.id IN $target_ids
            RETURN n.id, {
                name: n.name,
                type: n.type,
                properties: n.properties
            }, m.id, {
                name: m.name,
                type: m.type,
                properties: m.properties
            }, r.relationship_name, r.properties
            """

            result = await self.query(query, {"target_ids": target_ids})

            if not result:
                logger.info("No data returned for the supplied IDs")
                return [], []

            nodes_dict = {}
            edges = []

            for n_id, n_props, m_id, m_props, r_type, r_props_raw in result:
                if n_props.get("properties"):
                    try:
                        additional_props = json.loads(n_props["properties"])
                        n_props.update(additional_props)
                        del n_props["properties"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse properties JSON for node {n_id}")

                if m_props.get("properties"):
                    try:
                        additional_props = json.loads(m_props["properties"])
                        m_props.update(additional_props)
                        del m_props["properties"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse properties JSON for node {m_id}")

                nodes_dict[n_id] = (n_id, n_props)
                nodes_dict[m_id] = (m_id, m_props)

                edge_props = {}
                if r_props_raw:
                    try:
                        edge_props = json.loads(r_props_raw)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse edge properties for {n_id}->{m_id}")

                source_id = edge_props.get("source_node_id", n_id)
                target_id = edge_props.get("target_node_id", m_id)
                edges.append((source_id, target_id, r_type, edge_props))

            retrieval_time = time.time() - start_time
            logger.info(
                f"ID-filtered retrieval: {len(nodes_dict)} nodes and {len(edges)} edges in {retrieval_time:.2f}s"
            )

            return list(nodes_dict.values()), edges

        except Exception as e:
            logger.error(f"Error during ID-filtered graph data retrieval: {str(e)}")
            raise

    async def get_graph_metrics(self, include_optional=False) -> Dict[str, Any]:
        """
        Get metrics on graph structure and connectivity.

        This method computes various metrics around the graph, such as node and edge counts,
        mean degree, and connected component sizes. Optionally, it can include additional
        metrics based on user request.

        Parameters:
        -----------

            - include_optional: A boolean flag indicating whether to include optional metrics in
              the output. (default False)

        Returns:
        --------

            - Dict[str, Any]: A dictionary containing various metrics related to the graph.
        """

        try:
            node_count_result = await self.query("MATCH (n:Node) RETURN COUNT(n)")
            edge_count_result = await self.query("MATCH ()-[r:EDGE]->() RETURN COUNT(r)")
            num_nodes = node_count_result[0][0] if node_count_result else 0
            num_edges = edge_count_result[0][0] if edge_count_result else 0

            # Calculate mandatory metrics
            mandatory_metrics = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
                "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
                "num_connected_components": await self._get_num_connected_components(),
                "sizes_of_connected_components": await self._get_size_of_connected_components(),
            }

            if include_optional:
                # Calculate optional metrics
                shortest_path_lengths = await self._get_shortest_path_lengths()
                optional_metrics = {
                    "num_selfloops": await self._count_self_loops(),
                    "diameter": max(shortest_path_lengths) if shortest_path_lengths else -1,
                    "avg_shortest_path_length": sum(shortest_path_lengths)
                    / len(shortest_path_lengths)
                    if shortest_path_lengths
                    else -1,
                    "avg_clustering": await self._get_avg_clustering(),
                }
            else:
                optional_metrics = {
                    "num_selfloops": -1,
                    "diameter": -1,
                    "avg_shortest_path_length": -1,
                    "avg_clustering": -1,
                }

            return {**mandatory_metrics, **optional_metrics}

        except Exception as e:
            logger.error(f"Failed to get graph metrics: {e}")
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "mean_degree": 0,
                "edge_density": 0,
                "num_connected_components": 0,
                "sizes_of_connected_components": [],
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }

    async def _get_num_connected_components(self) -> int:
        """Get the number of connected components in the graph."""
        query = """
        MATCH (n:Node)
        WITH n, n.id AS node_id
        MATCH path = (n)-[:EDGE*1..3]-(m)
        WITH node_id, COLLECT(DISTINCT m.id) AS connected_nodes
        WITH COLLECT(DISTINCT connected_nodes + [node_id]) AS components
        RETURN SIZE(components) AS num_components
        """
        result = await self.query(query)
        return result[0][0] if result else 0

    async def _get_size_of_connected_components(self) -> List[int]:
        """Get the sizes of all connected components in the graph."""
        query = """
        MATCH (n:Node)
        WITH n, n.id AS node_id
        MATCH path = (n)-[:EDGE*1..3]-(m)
        WITH node_id, COLLECT(DISTINCT m.id) AS connected_nodes
        WITH COLLECT(DISTINCT connected_nodes + [node_id]) AS components
        UNWIND components AS component
        RETURN SIZE(component) AS component_size
        """
        result = await self.query(query)
        return [row[0] for row in result] if result else []

    async def _get_shortest_path_lengths(self) -> List[int]:
        """Get the lengths of shortest paths between all pairs of nodes."""
        query = """
        MATCH (n:Node), (m:Node)
        WHERE n.id < m.id
        MATCH path = (n)-[:EDGE*]-(m)
        RETURN MIN(LENGTH(path)) AS length
        """
        result = await self.query(query)
        return [row[0] for row in result if row[0] is not None] if result else []

    async def _count_self_loops(self) -> int:
        """Count the number of self-loops in the graph."""
        query = """
        MATCH (n:Node)-[r:EDGE]->(n)
        RETURN COUNT(r) AS count
        """
        result = await self.query(query)
        return result[0][0] if result else 0

    async def _get_avg_clustering(self) -> float:
        """Calculate the average clustering coefficient of the graph."""
        query = """
        MATCH (n:Node)-[:EDGE]-(neighbor)
        WITH n, COUNT(DISTINCT neighbor) as degree
        MATCH (n)-[:EDGE]-(n1)-[:EDGE]-(n2)-[:EDGE]-(n)
        WHERE n1 <> n2
        RETURN AVG(CASE WHEN degree <= 1 THEN 0 ELSE COUNT(DISTINCT n2) / (degree * (degree-1)) END) AS avg_clustering
        """
        result = await self.query(query)
        return result[0][0] if result and result[0][0] is not None else -1

    async def get_disconnected_nodes(self) -> List[str]:
        """
        Get nodes that are not connected to any other node.

        This method retrieves identifiers of nodes that lack any relationships in the graph,
        indicating they are standalone. It will return an empty list if no disconnected nodes
        exist.

        Returns:
        --------

            - List[str]: A list of identifiers for disconnected nodes.
        """
        query_str = """
        MATCH (n:Node)
        WHERE NOT EXISTS((n)-[]-())
        RETURN n.id
        """
        result = await self.query(query_str)
        return [str(row[0]) for row in result]

    # Graph Meta-Data Operations

    async def get_model_independent_graph_data(self) -> Dict[str, List[str]]:
        """
        Get graph data independent of any specific data model.

        This method returns a representation of the graph that includes distinct node labels and
        relationship types, making it easier to analyze the graph's structure without tying it
        to a specific implementation.

        Returns:
        --------

            - Dict[str, List[str]]: A dictionary summarizing the node labels and relationship
              types present in the graph.
        """
        node_labels = await self.query("MATCH (n:Node) RETURN DISTINCT labels(n)")
        rel_types = await self.query("MATCH ()-[r:EDGE]->() RETURN DISTINCT r.relationship_name")
        return {
            "node_labels": [label[0] for label in node_labels],
            "relationship_types": [rel[0] for rel in rel_types],
        }

    async def delete_graph(self) -> None:
        """
        Delete all data from the graph database.

        This method deletes all nodes and relationships from the graph database.
        It raises exceptions for failures occurring during deletion processes.
        """
        # In ``shared_ladybug_lock`` mode the Redis lock is the cross-process
        # mutex that ``query()`` uses to keep two processes from opening
        # the same Ladybug DB on disk concurrently. ``delete_graph`` removes
        # those files, so it MUST hold the same lock — otherwise a peer
        # process could be mid-query (holding the on-disk file lock) when
        # we delete its files, or could open the DB right between our
        # drop and our reopen.
        held_redis_lock = None
        if cache_config.shared_ladybug_lock and self.redis_lock is not None:
            held_redis_lock = await asyncio.to_thread(self.redis_lock.acquire_lock)
        try:
            # Transient drop: release the file handles so we can delete the
            # db files, but do NOT latch ``_permanently_closed`` — callers
            # expect to keep using this adapter after ``delete_graph`` and
            # have the store lazily reinitialize. Drain in-flight queries
            # under ``_connection_lock`` so we don't tear out a Connection
            # an executor thread is still using (the query path releases
            # ``_connection_lock`` before ``run_in_executor`` so multiple
            # queries can run concurrently — the lock alone wouldn't block
            # us against them).
            async with self._connection_lock:
                await self._drain_in_flight_queries()
                # Subprocess mode: ``_drop_native_resources`` issues two
                # RPCs (OP_CONN_CLOSE + OP_DB_CLOSE) which block on the
                # worker. Offload so we don't freeze the event loop.
                if self._subprocess_mode:
                    await asyncio.to_thread(self._drop_native_resources)
                else:
                    self._drop_native_resources()

            db_dir = os.path.dirname(self.db_path)
            db_name = os.path.basename(self.db_path)
            file_storage = get_file_storage(db_dir)

            if await file_storage.is_file(db_name):
                await file_storage.remove(db_name)
                await file_storage.remove(f"{db_name}.lock")
            else:
                await file_storage.remove_all(db_name)

            logger.info(f"Deleted Ladybug database files at {self.db_path}")

            # No eager reopen here: we just removed the on-disk store, and
            # callers asking us to delete it should not get a freshly
            # recreated empty database as a side effect. If the same
            # adapter is reused for a query later, ``get_or_init_connection``
            # rebuilds proxies + schema lazily — same shape local mode has
            # always had. If the caller is the dataset-deletion handler,
            # the cache evicts this entry and ``close()`` shuts the
            # subprocess down cleanly.

        except Exception as e:
            logger.error(f"Failed to delete graph data: {e}")
            raise
        finally:
            if held_redis_lock is not None:
                # Offloaded for symmetry with the acquire path; release
                # does Redis I/O too.
                await asyncio.to_thread(self.redis_lock.release_lock, held_redis_lock)

    async def get_document_subgraph(self, data_id: str):
        """
        Get all nodes that should be deleted when removing a document.

        This method constructs a complex query that identifies all nodes related to a specified
        document and returns a dictionary of these nodes. Ensures thorough checks for orphaned
        entities and inaccurate relationships that should be removed alongside the document.

        Parameters:
        -----------

            - data_id (str): The identifier for the document to query against.

        Returns:
        --------

            A dictionary containing details of the document and associated nodes that need to be
            deleted, or None if no related nodes are found.
        """
        query = """
        MATCH (doc:Node)
        WHERE (doc.type = 'TextDocument' OR doc.type = 'PdfDocument' OR doc.type = 'AudioDocument' OR doc.type = 'ImageDocument' OR doc.type = 'UnstructuredDocument') AND doc.id = $data_id

        OPTIONAL MATCH (doc)<-[e1:EDGE]-(chunk:Node)
        WHERE e1.relationship_name = 'is_part_of' AND chunk.type = 'DocumentChunk'

        OPTIONAL MATCH (chunk)-[e2:EDGE]->(entity:Node)
        WHERE e2.relationship_name = 'contains' AND entity.type = 'Entity'
        AND NOT EXISTS {
            MATCH (entity)<-[e3:EDGE]-(otherChunk:Node)-[e4:EDGE]->(otherDoc:Node)
            WHERE e3.relationship_name = 'contains'
            AND e4.relationship_name = 'is_part_of'
            AND (otherDoc.type = 'TextDocument' OR otherDoc.type = 'PdfDocument' OR otherDoc.type = 'AudioDocument' OR otherDoc.type = 'ImageDocument' OR otherDoc.type = 'UnstructuredDocument')
            AND otherDoc.id <> doc.id
        }

        OPTIONAL MATCH (chunk)<-[e5:EDGE]-(made_node:Node)
        WHERE e5.relationship_name = 'made_from' AND made_node.type = 'TextSummary'

        OPTIONAL MATCH (entity)-[e6:EDGE]->(type:Node)
        WHERE e6.relationship_name = 'is_a' AND type.type = 'EntityType'
        AND NOT EXISTS {
            MATCH (type)<-[e7:EDGE]-(otherEntity:Node)-[e8:EDGE]-(otherChunk:Node)-[e9:EDGE]-(otherDoc:Node)
            WHERE e7.relationship_name = 'is_a'
            AND e8.relationship_name = 'contains'
            AND e9.relationship_name = 'is_part_of'
            AND otherEntity.type = 'Entity'
            AND otherChunk.type = 'DocumentChunk'
            AND (otherDoc.type = 'TextDocument' OR otherDoc.type = 'PdfDocument' OR otherDoc.type = 'AudioDocument' OR otherDoc.type = 'ImageDocument' OR otherDoc.type = 'UnstructuredDocument')
            AND otherDoc.id <> doc.id
        }

        RETURN
            COLLECT(DISTINCT doc) as document,
            COLLECT(DISTINCT chunk) as chunks,
            COLLECT(DISTINCT entity) as orphan_entities,
            COLLECT(DISTINCT made_node) as made_from_nodes,
            COLLECT(DISTINCT type) as orphan_types
        """
        result = await self.query(query, {"data_id": f"{data_id}"})
        if not result or not result[0]:
            return None

        # Convert tuple to dictionary
        return {
            "document": result[0][0],
            "chunks": result[0][1],
            "orphan_entities": result[0][2],
            "made_from_nodes": result[0][3],
            "orphan_types": result[0][4],
        }

    async def get_degree_one_nodes(self, node_type: str):
        """
        Get all nodes that have only one connection.

        This method retrieves nodes which are connected to exactly one other node, identified by
        their specific type. It raises a ValueError if the input type is invalid and processes
        queries efficiently to return targeted results.

        Parameters:
        -----------

            - node_type (str): The type of nodes to filter by, must be 'Entity' or 'EntityType'.

        Returns:
        --------

            A list of nodes that have only one connection, as identified by the specified type.
        """
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        query = f"""
        MATCH (n:Node)
        WHERE n.type = '{node_type}'
        WITH n, COUNT {{ MATCH (n)--() }} as degree
        WHERE degree = 1
        RETURN n
        """
        result = await self.query(query)
        return [record[0] for record in result] if result else []

    def _normalize_temporal_ids(self, ids: Union[List[str], str]) -> List[str]:
        if isinstance(ids, str):
            return [uid.strip().strip("'\"") for uid in ids.split(",") if uid.strip()]

        return ids

    async def collect_events(self, ids: Union[List[str], str]) -> Any:
        """
        Collect all Event-type nodes reachable within 1..2 hops
        from the given node IDs.

        Args:
            graph_engine: Object exposing an async .query(str) -> Any
            ids: List of node IDs (strings)

        Returns:
            List of events
        """

        event_collection_cypher = """UNWIND $ids AS uid
            MATCH (start {id: uid})
            MATCH (start)-[*1..2]-(event)
            WHERE event.type = 'Event'
            WITH DISTINCT event
            RETURN collect(event) AS events;
        """

        ids = self._normalize_temporal_ids(ids)
        result = await self.query(event_collection_cypher, {"ids": ids})
        events = []
        if not result or not result[0] or not result[0][0]:
            return [{"events": events}]

        for node in result[0][0]:
            props = json.loads(node["properties"])

            event = {
                "id": node["id"],
                "name": node["name"],
                "description": props.get("description"),
            }

            if props.get("location"):
                event["location"] = props["location"]

            events.append(event)

        return [{"events": events}]

    async def collect_time_ids(
        self,
        time_from: Optional[Timestamp] = None,
        time_to: Optional[Timestamp] = None,
    ) -> List[str]:
        """
        Collect IDs of Timestamp nodes between time_from and time_to.

        Args:
            graph_engine: Object exposing an async .query(query, params) -> list[dict]
            time_from: Lower bound int (inclusive), optional
            time_to: Upper bound int (inclusive), optional

        Returns:
            A list of timestamp node IDs.
        """

        ids: List[str] = []

        if time_from and time_to:
            time_from = date_to_int(time_from)
            time_to = date_to_int(time_to)

            cypher = f"""
            MATCH (n:Node)
            WHERE n.type = 'Timestamp'
            // Extract time_at from the JSON string and cast to INT64
            WITH n, json_extract(n.properties, '$.time_at') AS t_str
            WITH n,
                 CASE
                   WHEN t_str IS NULL OR t_str = '' THEN NULL
                   ELSE CAST(t_str AS INT64)
                 END AS t
            WHERE t >= {time_from}
            AND t <= {time_to}
            RETURN n.id as id
            """

        elif time_from:
            time_from = date_to_int(time_from)

            cypher = f"""
            MATCH (n:Node)
            WHERE n.type = 'Timestamp'
            // Extract time_at from the JSON string and cast to INT64
            WITH n, json_extract(n.properties, '$.time_at') AS t_str
            WITH n,
                 CASE
                   WHEN t_str IS NULL OR t_str = '' THEN NULL
                   ELSE CAST(t_str AS INT64)
                 END AS t
            WHERE t >= {time_from}
            RETURN n.id as id
            """

        elif time_to:
            time_to = date_to_int(time_to)

            cypher = f"""
            MATCH (n:Node)
            WHERE n.type = 'Timestamp'
            // Extract time_at from the JSON string and cast to INT64
            WITH n, json_extract(n.properties, '$.time_at') AS t_str
            WITH n,
                 CASE
                   WHEN t_str IS NULL OR t_str = '' THEN NULL
                   ELSE CAST(t_str AS INT64)
                 END AS t
            WHERE t <= {time_to}
            RETURN n.id as id
            """

        else:
            return ids

        time_nodes = await self.query(cypher)
        time_ids_list = [item[0] for item in time_nodes]

        return time_ids_list

    async def get_triplets_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        """
        Retrieve a batch of triplets (start_node, relationship, end_node) from the graph.

        Parameters:
        -----------
            - offset (int): Number of triplets to skip before returning results.
            - limit (int): Maximum number of triplets to return.

        Returns:
        --------
            - list[dict[str, Any]]: A list of triplets, where each triplet is a dictionary
              with keys: 'start_node', 'relationship_properties', 'end_node'.

        Raises:
        -------
            - ValueError: If offset or limit are negative.
            - Exception: Re-raises any exceptions from query execution.
        """
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if limit < 0:
            raise ValueError(f"Limit must be non-negative, got {limit}")

        query = """
        MATCH (start_node:Node)-[relationship:EDGE]->(end_node:Node)
        RETURN {
            start_node: {
                id: start_node.id,
                name: start_node.name,
                type: start_node.type,
                properties: start_node.properties
            },
            relationship_properties: {
                relationship_name: relationship.relationship_name,
                properties: relationship.properties
            },
            end_node: {
                id: end_node.id,
                name: end_node.name,
                type: end_node.type,
                properties: end_node.properties
            }
        } AS triplet
        SKIP $offset LIMIT $limit
        """

        try:
            results = await self.query(query, {"offset": offset, "limit": limit})
        except Exception as e:
            logger.error(f"Failed to execute triplet query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: offset={offset}, limit={limit}")
            raise

        triplets = []
        for idx, row in enumerate(results):
            try:
                if not row or len(row) == 0:
                    logger.warning(f"Skipping empty row at index {idx} in triplet batch")
                    continue

                if not isinstance(row[0], dict):
                    logger.warning(
                        f"Skipping invalid row at index {idx}: expected dict, got {type(row[0])}"
                    )
                    continue

                triplet = row[0]

                if "start_node" not in triplet:
                    logger.warning(f"Skipping triplet at index {idx}: missing 'start_node' key")
                    continue

                if not isinstance(triplet["start_node"], dict):
                    logger.warning(f"Skipping triplet at index {idx}: 'start_node' is not a dict")
                    continue

                triplet["start_node"] = self._parse_node_properties(triplet["start_node"].copy())

                if "relationship_properties" not in triplet:
                    logger.warning(
                        f"Skipping triplet at index {idx}: missing 'relationship_properties' key"
                    )
                    continue

                if not isinstance(triplet["relationship_properties"], dict):
                    logger.warning(
                        f"Skipping triplet at index {idx}: 'relationship_properties' is not a dict"
                    )
                    continue

                rel_props = triplet["relationship_properties"].copy()
                relationship_name = rel_props.get("relationship_name") or ""

                if rel_props.get("properties"):
                    try:
                        parsed_props = json.loads(rel_props["properties"])
                        if isinstance(parsed_props, dict):
                            rel_props.update(parsed_props)
                            del rel_props["properties"]
                        else:
                            logger.warning(
                                f"Parsed relationship properties is not a dict for triplet at index {idx}"
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"Failed to parse relationship properties JSON for triplet at index {idx}: {e}"
                        )

                rel_props["relationship_name"] = relationship_name
                triplet["relationship_properties"] = rel_props

                if "end_node" not in triplet:
                    logger.warning(f"Skipping triplet at index {idx}: missing 'end_node' key")
                    continue

                if not isinstance(triplet["end_node"], dict):
                    logger.warning(f"Skipping triplet at index {idx}: 'end_node' is not a dict")
                    continue

                triplet["end_node"] = self._parse_node_properties(triplet["end_node"].copy())

                triplets.append(triplet)

            except Exception as e:
                logger.error(f"Error processing triplet at index {idx}: {e}", exc_info=True)
                continue

        return triplets
