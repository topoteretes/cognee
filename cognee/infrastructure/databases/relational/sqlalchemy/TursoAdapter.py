"""Relational adapter for Turso (libSQL), a drop-in replacement for SQLite.

A libSQL database file is a SQLite file, so cognee talks to Turso through the
same aiosqlite driver, sqlite dialect, and sqlite-dialect Alembic migrations it
already uses for the SQLite backend. ``TursoAdapter`` is therefore a thin
subclass of :class:`SQLAlchemyAdapter` that inherits all of its behavior
unchanged.

Modes
-----
* **Local / embedded** — a libSQL file on disk. Identical to the SQLite backend;
  the adapter adds nothing (all the machinery below is a no-op).
* **Remote** — Turso's embedded-replica model. cognee reads and writes a fast
  local replica through aiosqlite exactly as in local mode; ``libsql-experimental``
  keeps that replica in sync with the hosted primary. The replica is seeded from
  the primary before first use, and each write is followed by a sync so changes
  reach the cloud within the operation (cognee runs many flows as a one-shot
  ``asyncio.run``, so a background timer could be cancelled before it ever fired).

Note: writes are applied to the replica through aiosqlite. Whether libSQL's sync
propagates those to the primary depends on the driver's replica write-capture and
must be confirmed against a live Turso database; the local drop-in path is fully
exercised offline.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.shared.logging_utils import get_logger

from .SqlAlchemyAdapter import SQLAlchemyAdapter

logger = get_logger()


class TursoAdapter(SQLAlchemyAdapter):
    """Relational adapter for Turso (libSQL); a SQLite drop-in backed by aiosqlite."""

    def __init__(
        self,
        connection_string: str,
        connect_args: dict = None,
        pool_args: dict = None,
        sync_url: str = None,
        auth_token: str = None,
    ):
        # aiosqlite drives the query path in both modes, so the base adapter's
        # sqlite branch builds the async engine, WAL pragmas and sessionmaker —
        # every inherited method then works unchanged. That is the drop-in.
        super().__init__(connection_string, connect_args=connect_args, pool_args=pool_args)

        # Remote mode keeps the local replica (``self.db_path``) in sync with a
        # hosted Turso primary; local mode leaves these unset and adds nothing.
        self._sync_url = sync_url
        self._auth_token = auth_token
        self._sync_lock = asyncio.Lock()
        self._seeded = False

    @property
    def is_remote(self) -> bool:
        return self._sync_url is not None

    def _sync_replica(self) -> None:
        """Open a short-lived embedded-replica connection and sync it with the primary.

        Blocking (libsql-experimental is synchronous), so it is always run via
        ``asyncio.to_thread``. Each sync uses its own connection, so it never
        holds a second handle to the replica file between syncs and is safe to
        run from a worker thread.
        """
        import libsql_experimental as libsql

        # The base adapter creates the databases directory lazily (in
        # create_database), so ensure it exists before opening the replica.
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        connection = libsql.connect(
            self.db_path, sync_url=self._sync_url, auth_token=self._auth_token
        )
        try:
            connection.sync()
        finally:
            connection.close()

    async def sync(self) -> None:
        """Sync the local replica with the remote primary; a no-op in local mode.

        Best-effort: a transient sync failure is logged, not raised, so it never
        breaks a database operation — the local replica stays usable and the next
        sync retries.
        """
        if not self.is_remote:
            return
        async with self._sync_lock:
            try:
                await asyncio.to_thread(self._sync_replica)
            except Exception:
                logger.warning("Turso replica sync failed", exc_info=True)

    async def _seed_from_primary(self) -> None:
        """Pull the primary's current state into the replica once, before first use."""
        if self.is_remote and not self._seeded:
            self._seeded = True
            await self.sync()

    async def _write(self, method, *args, **kwargs):
        """Run a base-class write, then push the change to the primary (remote mode).

        Syncing right after the write — rather than on a background timer — keeps
        the push inside the caller's operation, so it still happens when cognee
        runs a flow as a one-shot ``asyncio.run`` that tears the loop down
        immediately afterwards.
        """
        await self._seed_from_primary()
        result = await method(self, *args, **kwargs)
        await self.sync()
        return result

    async def create_table(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.create_table, *args, **kwargs)

    async def delete_table(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.delete_table, *args, **kwargs)

    async def create_database(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.create_database, *args, **kwargs)

    async def insert_data(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.insert_data, *args, **kwargs)

    async def delete_entity_by_id(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.delete_entity_by_id, *args, **kwargs)

    async def delete_data_entity(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.delete_data_entity, *args, **kwargs)

    async def drop_tables(self, *args, **kwargs):
        return await self._write(SQLAlchemyAdapter.drop_tables, *args, **kwargs)

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        # Seed the replica from the primary before the first read (remote mode).
        await self._seed_from_primary()
        async with SQLAlchemyAdapter.get_async_session(self) as session:
            yield session
