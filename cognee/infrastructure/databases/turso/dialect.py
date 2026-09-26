"""SQLAlchemy dialect cognee uses to talk to the Turso rewrite engine.

``pyturso`` ships ``sqlite+aioturso://``, an asyncio dialect over ``turso.aio``.
Two of its details do not fit cognee, so this module registers a subclass as
``sqlite+cognee_turso://``:

* SQLAlchemy 2.0.4x+ asks the async DBAPI adapter for ``has_stop`` (whether the
  driver connection has a ``stop()`` to terminate it). ``AsyncAdapt_turso_dbapi``
  does not define it, so ``create_async_engine`` fails before the first query.
* The upstream reflection mixin returns empty lists for indexes, foreign keys and
  unique constraints, claiming ``PRAGMA index_list`` / ``foreign_key_list`` are
  unsupported. The engine answers both PRAGMAs, and cognee relies on reflection:
  idempotent migrations check for an existing index before creating it, and
  Alembic's ``batch_alter_table`` re-creates a table from what it reflects, so
  stubbed reflection would silently drop foreign keys and unique constraints.
  The stock SQLite implementations are restored here.

The dialect also drops the ``aiosqlite`` connect arguments cognee's SQLite branch
passes (``timeout``, ``check_same_thread``), which ``turso.aio.connect`` rejects;
``PRAGMA busy_timeout`` covers the wait. Its statement compiler
(:mod:`cognee.infrastructure.databases.turso.compiler`) flattens nested joins, which
the engine rejects in their parenthesized form.
"""

from __future__ import annotations

from sqlalchemy.dialects import registry
from sqlalchemy.dialects.sqlite.aiosqlite import SQLiteDialect_aiosqlite

from .compiler import CogneeTursoCompiler
from .runtime import INSTALL_HINT

DIALECT_NAME = "sqlite"
DRIVER_NAME = "cognee_turso"

# Connect arguments the SQLite/aiosqlite code paths pass that turso.aio.connect
# does not accept. Dropped rather than forwarded so both branches share one config.
_UNSUPPORTED_CONNECT_ARGS = ("timeout", "check_same_thread", "uri", "detect_types")


try:
    from turso.sqlalchemy.dialect import AioTursoDialect, AsyncAdapt_turso_dbapi
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(INSTALL_HINT) from error


class CogneeTursoDialect(AioTursoDialect):
    """``sqlite+cognee_turso://`` — pyturso's asyncio dialect with cognee's fixes."""

    name = DIALECT_NAME
    driver = DRIVER_NAME
    supports_statement_cache = True

    # Stock SQLite reflection. The engine supports the PRAGMAs these use.
    get_foreign_keys = SQLiteDialect_aiosqlite.get_foreign_keys
    get_indexes = SQLiteDialect_aiosqlite.get_indexes
    get_unique_constraints = SQLiteDialect_aiosqlite.get_unique_constraints
    get_check_constraints = SQLiteDialect_aiosqlite.get_check_constraints
    get_multi_indexes = SQLiteDialect_aiosqlite.get_multi_indexes
    get_multi_unique_constraints = SQLiteDialect_aiosqlite.get_multi_unique_constraints
    get_multi_foreign_keys = SQLiteDialect_aiosqlite.get_multi_foreign_keys
    get_multi_check_constraints = SQLiteDialect_aiosqlite.get_multi_check_constraints
    get_temp_table_names = SQLiteDialect_aiosqlite.get_temp_table_names
    get_temp_view_names = SQLiteDialect_aiosqlite.get_temp_view_names

    # The engine reports SQLite 3.50 but rejects a parenthesized join in a FROM
    # clause, which SQLAlchemy emits for joined-table inheritance targets and
    # nested eager loads; this compiler renders those joins as a flat chain.
    statement_compiler = CogneeTursoCompiler

    @classmethod
    def import_dbapi(cls):
        import turso
        import turso.aio

        dbapi = AsyncAdapt_turso_dbapi(turso.aio, turso)
        # turso.aio.Connection has no stop(); SQLAlchemy then disables
        # terminate() and closes connections the ordinary way.
        dbapi.has_stop = False
        return dbapi

    def connect(self, *cargs, **cparams):
        for key in _UNSUPPORTED_CONNECT_ARGS:
            cparams.pop(key, None)
        return super().connect(*cargs, **cparams)


_registered = False


def register_dialect() -> None:
    """Make ``sqlite+cognee_turso://`` resolvable by ``create_async_engine``. Idempotent."""
    global _registered
    if not _registered:
        registry.register(f"{DIALECT_NAME}.{DRIVER_NAME}", __name__, "CogneeTursoDialect")
        _registered = True
