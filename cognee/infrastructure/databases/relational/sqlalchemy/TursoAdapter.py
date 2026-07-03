"""Relational adapter for Turso (libSQL) on the aioturso async engine.

Turso is a SQLite-compatible database with a from-scratch, async-first Rust engine
(shipped as the ``pyturso`` package, imported as ``turso``). For local/embedded use
a libSQL database file is byte-compatible with SQLite, so this adapter reuses all of
:class:`SQLAlchemyAdapter`'s behavior unchanged and only bridges one packaging skew.

Why we can subclass with almost no overrides: the connection string uses the
``sqlite+aioturso://`` dialect, so every place the base adapter branches on
``"sqlite" in connection_string`` or ``dialect.name == "sqlite"`` (PRAGMA
``foreign_keys=ON``, schema-less ``DROP TABLE``, ``metadata.reflect``) behaves
exactly as it does for the built-in sqlite backend, and the sqlite-dialect Alembic
migrations apply unchanged.

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
import random

from sqlalchemy.exc import DBAPIError, OperationalError

from .SqlAlchemyAdapter import SQLAlchemyAdapter

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

    def __init__(self, connection_string: str, connect_args: dict = None, pool_args: dict = None):
        ensure_turso_dialect_compatibility()
        super().__init__(connection_string, connect_args=connect_args, pool_args=pool_args)
        # Serializes this adapter's writes against other in-process async tasks so a
        # single-writer libSQL database never sees two concurrent writers from cognee.
        self._write_lock = asyncio.Lock()

    async def _run_write(self, method, *args, **kwargs):
        """Run a base-class write method serialized by the write lock, retrying busy errors.

        The lock removes in-process contention entirely (the common case: two async
        tasks writing at once). The retry loop covers residual busy errors that can
        still arrive from another process or the remote primary.
        """
        attempt = 0
        while True:
            try:
                async with self._write_lock:
                    return await method(self, *args, **kwargs)
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
