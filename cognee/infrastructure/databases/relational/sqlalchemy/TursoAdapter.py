"""Relational adapter for Turso (libSQL) on the aioturso async engine.

Turso is a SQLite-compatible database with a from-scratch, async-first Rust engine
(shipped as the ``pyturso`` package, imported as ``turso``). This adapter supports two
modes, both driven by the ``sqlite+aioturso://`` dialect so the base
:class:`SQLAlchemyAdapter` behavior (PRAGMA ``foreign_keys=ON``, schema-less
``DROP TABLE``, ``metadata.reflect``) and the sqlite-dialect Alembic migrations apply
unchanged, and so all existing ``get_async_session`` call sites keep working:

* **Local / embedded** — a libSQL file, byte-compatible with SQLite. Reuses the base
  adapter with almost no overrides.
* **Remote** — a local *embedded replica* bound to a remote Turso database. Reads and
  writes hit the fast local replica (real async, no thread-queue), and a lightweight
  background task syncs it with the remote primary (``push`` local writes up,
  ``pull`` remote changes down). This is Turso's recommended production model.

Write conflicts
---------------
Turso/libSQL is single-writer, so two async tasks writing concurrently race: under
load most attempts fail with a "database is locked"/busy error and read-modify-write
sequences can lose updates. On the Rust engine ``PRAGMA busy_timeout`` is not honored
(verified), so the classic SQLite fix does not apply. We therefore serialize the
adapter's own writes within the process with an ``asyncio.Lock`` (the same approach
:class:`PGVectorAdapter` uses for its write paths) and retry transient busy errors
with bounded backoff for cross-process / remote-primary contention.
"""

import asyncio
import os
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import NullPool
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cognee.shared.logging_utils import get_logger

from .SqlAlchemyAdapter import SQLAlchemyAdapter

logger = get_logger()

# Substrings that mark a transient single-writer contention error worth retrying.
_BUSY_MARKERS = ("database is locked", "busy", "sqlite_busy", "write-write conflict")


def _is_busy_error(error: BaseException) -> bool:
    return any(marker in str(error).lower() for marker in _BUSY_MARKERS)


def ensure_turso_dialect_compatibility() -> None:
    """Bridge a version skew between pyturso's aioturso dialect and SQLAlchemy 2.0.x.

    pyturso's ``AioTursoDialect`` subclasses SQLAlchemy's built-in aiosqlite dialect,
    whose ``__init__`` reads ``dbapi.has_stop`` (a flag SQLAlchemy sets for aiosqlite
    to know whether the connection exposes a ``stop()`` for clean worker-thread
    shutdown). pyturso 0.6.1 does not set it on its DBAPI adapter, so building an
    engine raises ``AttributeError: ... has no attribute 'has_stop'``. We set it to
    ``False`` (the safe default: skip the optional stop path). The same one-line fix
    is being contributed upstream to pyturso.
    """
    from turso.sqlalchemy.dialect import AsyncAdapt_turso_dbapi

    if not hasattr(AsyncAdapt_turso_dbapi, "has_stop"):
        AsyncAdapt_turso_dbapi.has_stop = False


class TursoAdapter(SQLAlchemyAdapter):
    """SQLAlchemy relational adapter backed by the aioturso (Rust) async driver."""

    # aioturso's connect() takes (database, *, experimental_features, isolation_level,
    # extra_io) and rejects aiosqlite's `timeout` arg, so drop the sqlite default.
    _sqlite_default_connect_args: dict = {}

    # Bounded retry for transient busy errors (cross-process / remote primary).
    _write_max_retries: int = 5
    _write_retry_base_delay: float = 0.02

    def __init__(
        self,
        connection_string: str = None,
        connect_args: dict = None,
        pool_args: dict = None,
        turso_url: str = None,
        auth_token: str = None,
        replica_path: str = None,
        sync_interval: float = 2.0,
    ):
        ensure_turso_dialect_compatibility()
        # Serializes this adapter's writes against other in-process async tasks so a
        # single-writer libSQL database never sees two concurrent writers from cognee.
        self._write_lock = asyncio.Lock()

        self._is_remote = bool(turso_url)
        if self._is_remote:
            self._init_remote(
                replica_path, turso_url, auth_token, connect_args or {}, sync_interval
            )
        else:
            super().__init__(connection_string, connect_args=connect_args, pool_args=pool_args)

    # ── Remote (embedded-replica) setup ────────────────────────────────────────
    def _init_remote(self, replica_path, turso_url, auth_token, connect_args, sync_interval):
        import turso.aio.sync

        os.makedirs(os.path.dirname(replica_path), exist_ok=True)
        self._turso_url = turso_url
        self._auth_token = auth_token
        self._replica_path = replica_path
        self._sync_interval = sync_interval
        self._sync_conn = None
        self._sync_task = None
        self._sync_lock = asyncio.Lock()

        self.db_path = replica_path
        self.db_uri = f"sqlite+aioturso://embedded-replica/{turso_url}"

        def _make_conn(*_args, **_kwargs):
            # The aioturso dialect awaits this to obtain its DBAPI connection; we hand
            # it an embedded-replica connection bound to the remote primary.
            return turso.aio.sync.connect(replica_path, turso_url, auth_token=auth_token)

        self.engine = create_async_engine(
            "sqlite+aioturso://",
            poolclass=NullPool,
            connect_args={**connect_args, "async_creator_fn": _make_conn},
        )
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    async def _get_sync_conn(self):
        import turso.aio.sync

        if self._sync_conn is None:
            self._sync_conn = await turso.aio.sync.connect(
                self._replica_path, self._turso_url, auth_token=self._auth_token
            )
        return self._sync_conn

    async def sync(self) -> None:
        """Push local replica writes to the remote primary and pull remote changes.

        Sync is file-level (a push from any connection flushes all committed local
        writes), so one dedicated connection covers writes made by every pooled
        session. Serialized so overlapping ticks/flushes never sync concurrently.
        """
        if not self._is_remote:
            return
        async with self._sync_lock:
            conn = await self._get_sync_conn()
            await conn.push()
            await conn.pull()

    def _ensure_sync_loop(self) -> None:
        """Start the background sync loop lazily, on the running event loop."""
        if self._is_remote and self._sync_task is None:
            self._sync_task = asyncio.ensure_future(self._sync_loop())

    async def _sync_loop(self) -> None:
        while True:
            await asyncio.sleep(self._sync_interval)
            try:
                await self.sync()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Turso background replica sync failed", exc_info=True)

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        # Kick off the background replica sync on first use (needs a running loop).
        self._ensure_sync_loop()
        async with SQLAlchemyAdapter.get_async_session(self) as session:
            yield session

    # ── Write serialization + retry (both modes) ───────────────────────────────
    async def _run_write(self, method, *args, **kwargs):
        """Run a base-class write method serialized by the write lock, retrying busy errors.

        The lock removes in-process contention entirely (the common case: two async
        tasks writing at once). The retry loop covers residual busy errors that can
        still arrive from another process or the remote primary. In remote mode the
        write is flushed to the primary right after it succeeds.
        """
        attempt = 0
        while True:
            try:
                async with self._write_lock:
                    result = await method(self, *args, **kwargs)
                if self._is_remote:
                    await self.sync()
                return result
            except (DBAPIError, OperationalError) as error:
                attempt += 1
                if attempt > self._write_max_retries or not _is_busy_error(error):
                    raise
                # Exponential backoff with jitter before retrying.
                await asyncio.sleep(self._write_retry_base_delay * (2**attempt) * random.random())

    async def create_table(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.create_table, *args, **kwargs)

    async def delete_table(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.delete_table, *args, **kwargs)

    async def insert_data(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.insert_data, *args, **kwargs)

    async def delete_entity_by_id(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.delete_entity_by_id, *args, **kwargs)

    async def delete_data_entity(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.delete_data_entity, *args, **kwargs)

    async def drop_tables(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.drop_tables, *args, **kwargs)

    async def delete_database(self, *args, **kwargs):
        return await self._run_write(SQLAlchemyAdapter.delete_database, *args, **kwargs)
