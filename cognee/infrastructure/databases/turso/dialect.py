"""SQLAlchemy dialect cognee uses to talk to the Turso rewrite engine.

``pyturso`` ships ``sqlite+aioturso://``, an asyncio dialect over ``turso.aio``.
One of its details does not fit cognee, so this module registers a subclass as
``sqlite+cognee_turso://``: SQLAlchemy 2.0.4x+ asks the async DBAPI adapter for
``has_stop`` (whether the driver connection has a ``stop()`` to terminate it).
``AsyncAdapt_turso_dbapi`` does not define it, so ``create_async_engine`` fails
before the first query.

The dialect also drops the ``aiosqlite`` connect arguments cognee's SQLite branch
passes (``timeout``, ``check_same_thread``), which ``turso.aio.connect`` rejects;
``PRAGMA busy_timeout`` covers the wait. Its statement compiler
(:mod:`cognee.infrastructure.databases.turso.compiler`) flattens nested joins, which
the engine rejects in their parenthesized form.
"""

from __future__ import annotations

from sqlalchemy.dialects import registry

from .compiler import CogneeTursoCompiler
from .runtime import INSTALL_HINT, explain_file_in_use

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
        # The engine locks a file to one process; say so in cognee's terms. A
        # non-DBAPI error is not re-wrapped by SQLAlchemy, so callers see it as is.
        database_path = cargs[0] if cargs else cparams.get("database", "")
        with explain_file_in_use(database_path):
            try:
                return super().connect(*cargs, **cparams)
            except Exception as error:
                # pyturso neither exports nor maps its IoError (e.g. a missing directory),
                # so SQLAlchemy would pass it through raw. Raise it as the DBAPI error
                # stock sqlite3 raises, so callers catching OperationalError handle it.
                if type(error).__name__ != "IoError":
                    raise
                import turso

                raise turso.OperationalError(f"unable to open database file: {error}") from error


_registered = False


def register_dialect() -> None:
    """Make ``sqlite+cognee_turso://`` resolvable by ``create_async_engine``. Idempotent."""
    global _registered
    if not _registered:
        registry.register(f"{DIALECT_NAME}.{DRIVER_NAME}", __name__, "CogneeTursoDialect")
        _registered = True
