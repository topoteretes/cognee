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
"""

from .SqlAlchemyAdapter import SQLAlchemyAdapter


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

    def __init__(self, connection_string: str, connect_args: dict = None, pool_args: dict = None):
        ensure_turso_dialect_compatibility()
        super().__init__(connection_string, connect_args=connect_args, pool_args=pool_args)
