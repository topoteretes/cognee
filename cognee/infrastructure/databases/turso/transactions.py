"""Connection setup, transaction mode and conflict retries for Turso engines.

Every Turso-backed SQLAlchemy engine goes through :func:`configure_engine`, which
applies the connection PRAGMAs (journal mode from ``TursoConfig``, ``synchronous``,
``busy_timeout``, optional ``foreign_keys``) on each new connection.

In ``mvcc`` mode the engine additionally opens transactions itself: connections are
created in driver autocommit (``isolation_level=None``, see
:func:`connect_args_for_mode`) and a ``begin`` listener emits ``BEGIN CONCURRENT``,
so ordinary writes commit in parallel. DDL needs an exclusive transaction under
MVCC ("DDL statements require an exclusive transaction"), so schema work runs inside
:func:`exclusive_transaction`, which flips a context variable the listener reads and
emits a plain ``BEGIN`` instead. The variable is a ``ContextVar``: SQLAlchemy runs
the sync listener in a greenlet that inherits the awaiting task's context.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

from cognee.shared.logging_utils import get_logger

from .config import TursoConfig, get_turso_config

logger = get_logger()

T = TypeVar("T")

_exclusive_ddl: ContextVar[bool] = ContextVar("cognee_turso_exclusive_ddl", default=False)

# Substrings of the driver messages that mean "retry the whole transaction".
_RETRYABLE_MARKERS = ("write-write conflict", "conflict", "busy", "database is locked")


def connect_pragmas(config: TursoConfig | None = None, *, foreign_keys: bool = False) -> list[str]:
    """PRAGMA statements to run on every new connection, in order."""
    config = config or get_turso_config()
    statements = [
        f"PRAGMA journal_mode={config.turso_journal_mode}",
        "PRAGMA synchronous=NORMAL",
        f"PRAGMA busy_timeout={config.turso_busy_timeout_ms}",
    ]
    if foreign_keys:
        # SQLite disables FK enforcement per connection by default.
        statements.insert(0, "PRAGMA foreign_keys=ON")
    return statements


def connect_args_for_mode(config: TursoConfig | None = None) -> dict[str, Any]:
    """Driver connect kwargs the journal mode requires.

    MVCC needs driver autocommit so the ``begin`` listener controls the transaction
    statement; WAL keeps the driver's implicit ``BEGIN DEFERRED``.
    """
    config = config or get_turso_config()
    return {"isolation_level": None} if config.concurrent_writes else {}


def apply_pragmas(dbapi_connection, statements: Iterable[str]) -> None:
    """Run ``statements`` on a DB-API connection, stepping each so it takes effect.

    pyturso prepares a statement on ``execute`` and runs it when the cursor is
    read, so a PRAGMA whose result is never fetched may never apply.
    """
    cursor = dbapi_connection.cursor()
    try:
        for statement in statements:
            cursor.execute(statement)
            cursor.fetchall()
    finally:
        cursor.close()


def install_connect_pragmas(engine: AsyncEngine, statements: Iterable[str]) -> None:
    """Run ``statements`` on every new DB-API connection of ``engine``."""
    statements = list(statements)

    @event.listens_for(engine.sync_engine, "connect")
    def _turso_connect(dbapi_connection, _record):
        apply_pragmas(dbapi_connection, statements)


def install_transaction_hook(engine: AsyncEngine, config: TursoConfig | None = None) -> None:
    """In mvcc mode, open transactions with ``BEGIN CONCURRENT`` (plain ``BEGIN`` for DDL).

    Requires the engine's connections to be in driver autocommit (see
    :func:`connect_args_for_mode`). A no-op in WAL mode.
    """
    config = config or get_turso_config()
    if not config.concurrent_writes:
        return

    @event.listens_for(engine.sync_engine, "begin")
    def _turso_begin(connection):
        connection.exec_driver_sql("BEGIN" if _exclusive_ddl.get() else "BEGIN CONCURRENT")


def configure_engine(
    engine: AsyncEngine, *, foreign_keys: bool = False, config: TursoConfig | None = None
) -> None:
    """Install cognee's connection PRAGMAs and (in mvcc mode) the transaction hook."""
    config = config or get_turso_config()
    install_connect_pragmas(engine, connect_pragmas(config, foreign_keys=foreign_keys))
    install_transaction_hook(engine, config)


def begin_statement(config: TursoConfig | None = None, *, ddl: bool = False) -> str | None:
    """Statement a raw DB-API caller opens a transaction with, or None to rely on the driver."""
    config = config or get_turso_config()
    if not config.concurrent_writes:
        return None
    return "BEGIN" if ddl else "BEGIN CONCURRENT"


@asynccontextmanager
async def exclusive_transaction() -> AsyncIterator[None]:
    """Run schema changes inside; MVCC engines then open plain ``BEGIN`` transactions."""
    token = _exclusive_ddl.set(True)
    try:
        yield
    finally:
        _exclusive_ddl.reset(token)


def is_retryable_conflict(error: BaseException) -> bool:
    """True for the engine errors that a fresh attempt of the same transaction may clear."""
    message = str(error).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


async def retry_on_conflict(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int | None = None,
    base_delay: float = 0.005,
) -> T:
    """Run ``operation`` (one whole transaction) again on a write conflict.

    MVCC detects two transactions writing the same row eagerly and aborts the later
    one with ``Write-write conflict``; under WAL a writer that outlives
    ``busy_timeout`` fails with ``database is locked``. Both are transient, so the
    transaction is re-run with jittered backoff, ``TURSO_CONFLICT_RETRIES`` times.
    Any other error propagates unchanged, as does the last conflict.
    """
    if attempts is None:
        attempts = get_turso_config().turso_conflict_retries
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as error:
            if attempt >= attempts or not is_retryable_conflict(error):
                raise
            attempt += 1
            delay = base_delay * (2**attempt) * (0.5 + random.random())
            logger.debug(
                "Turso write conflict, retrying (%d/%d) in %.3fs: %s",
                attempt,
                attempts,
                delay,
                str(error).splitlines()[0],
            )
            await asyncio.sleep(delay)
