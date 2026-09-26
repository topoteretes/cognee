"""Relational adapter for Turso, the Rust rewrite of SQLite (``pyturso``).

A Turso database is a SQLite-compatible file, so cognee keeps the sqlite dialect,
the sqlite-dialect Alembic migrations and every inherited
:class:`SQLAlchemyAdapter` method. What changes is the driver: the engine talks to
the Turso rewrite through ``turso.aio`` via cognee's ``sqlite+cognee_turso://``
dialect (:mod:`cognee.infrastructure.databases.turso.dialect`) instead of
``aiosqlite``.

Journal mode and timeouts come from :class:`TursoConfig` (``TURSO_*`` env vars).
In ``mvcc`` mode ordinary transactions run as ``BEGIN CONCURRENT`` and the schema
operations below run inside :func:`exclusive_transaction`, since DDL needs a plain
``BEGIN`` under MVCC.

Only local files are supported. Remote Turso databases (``DB_TURSO_URL``) are
rejected by ``create_relational_engine`` in this version.
"""

from cognee.infrastructure.databases.turso import (
    configure_engine,
    connect_args_for_mode,
    exclusive_transaction,
    get_turso_config,
    remove_database_files,
    turso_url,
)
from cognee.shared.logging_utils import get_logger

from .SqlAlchemyAdapter import SQLAlchemyAdapter

logger = get_logger()


class TursoAdapter(SQLAlchemyAdapter):
    """Relational adapter for a local Turso database file on the rewrite engine."""

    def __init__(
        self,
        database_path: str,
        connect_args: dict | None = None,
        pool_args: dict | None = None,
    ):
        self.turso_config = get_turso_config()
        connect_args = {**(connect_args or {}), **connect_args_for_mode(self.turso_config)}
        # The base adapter's sqlite branch builds the async engine and the
        # sessionmaker and calls _configure_sqlite_engine (below) for the
        # per-connection setup.
        super().__init__(turso_url(database_path), connect_args=connect_args, pool_args=pool_args)

    def _configure_sqlite_engine(self) -> None:
        # The one Turso engine policy, shared with the graph and cache engines:
        # journal mode + timeouts on every connection, BEGIN CONCURRENT in mvcc.
        configure_engine(self.engine, config=self.turso_config)

    # DDL must run in an exclusive transaction under MVCC; a no-op in WAL mode.

    async def create_database(self, *args, **kwargs):
        async with exclusive_transaction():
            return await super().create_database(*args, **kwargs)

    async def create_table(self, *args, **kwargs):
        async with exclusive_transaction():
            return await super().create_table(*args, **kwargs)

    async def delete_table(self, *args, **kwargs):
        async with exclusive_transaction():
            return await super().delete_table(*args, **kwargs)

    async def drop_tables(self, *args, **kwargs):
        async with exclusive_transaction():
            return await super().drop_tables(*args, **kwargs)

    async def delete_database(self):
        """Remove the database file and the engine's companion files.

        The base method removes the main file; the WAL/MVCC companions
        (``-wal``, ``-shm``, ``-log``) would otherwise survive and be picked up by
        a same-name database created later.
        """
        await super().delete_database()
        if not self.db_path:
            return
        try:
            remove_database_files(self.db_path)
        except OSError as error:
            # Best effort, like the base class's own file removal: a lingering
            # companion must never poison teardown.
            logger.warning("Could not remove Turso database files for %s: %s", self.db_path, error)
