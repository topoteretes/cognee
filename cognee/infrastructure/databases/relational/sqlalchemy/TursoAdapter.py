"""Relational adapter for Turso, the Rust rewrite of SQLite (``pyturso``).

A Turso database is a SQLite-compatible file, so cognee keeps the sqlite dialect,
the sqlite-dialect Alembic migrations and every inherited
:class:`SQLAlchemyAdapter` method. What changes is the driver: the engine talks to
the Turso rewrite through ``turso.aio`` via cognee's ``sqlite+cognee_turso://``
dialect (:mod:`cognee.infrastructure.databases.turso.dialect`) instead of
``aiosqlite``.

Timeouts come from :class:`TursoConfig` (``TURSO_*`` env vars), but the journal
mode is always ``wal``, whatever ``TURSO_JOURNAL_MODE`` says: ORM sessions cannot
be re-run after an ``mvcc`` write-write conflict (see :meth:`TursoConfig.wal_only`).

Only local files are supported. Remote Turso databases (``DB_TURSO_URL``) are
rejected by ``create_relational_engine`` in this version.
"""

from cognee.infrastructure.databases.turso import (
    configure_engine,
    connect_args_for_mode,
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
        self.turso_config = get_turso_config().wal_only()
        connect_args = {**(connect_args or {}), **connect_args_for_mode(self.turso_config)}
        # The base adapter's sqlite branch builds the async engine and the
        # sessionmaker and calls _configure_sqlite_engine (below) for the
        # per-connection setup.
        super().__init__(turso_url(database_path), connect_args=connect_args, pool_args=pool_args)

    def _configure_sqlite_engine(self) -> None:
        # The one Turso engine policy, shared with the graph and cache engines:
        # journal mode + timeouts on every connection. The config is pinned to
        # wal, so no BEGIN CONCURRENT hook is installed and DDL needs no
        # exclusive_transaction().
        configure_engine(self.engine, config=self.turso_config)

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
