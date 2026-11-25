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

Engines with read-modify-write paths (graph, vector) need the same control in
``wal`` mode: the driver's implicit ``BEGIN`` is only issued before the first
INSERT/UPDATE/DELETE, so a SELECT that precedes the write runs outside the
transaction and a concurrent commit in between is silently overwritten. Such
engines opt in with ``immediate_writes=True``; their writes then run inside
:func:`write_transaction`, which makes the listener emit ``BEGIN IMMEDIATE``. That
takes the write lock before the first read, so a second writer waits on
``busy_timeout`` and then reads the first one's commit.
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
_write_txn: ContextVar[bool] = ContextVar("cognee_turso_write_transaction", default=False)

# The engine's own messages for a transient write collision, matched exactly (lowercased):
#   "Write-write conflict"  -- mvcc: two BEGIN CONCURRENT transactions wrote the same row
#   "database is locked"    -- wal: a writer outlived busy_timeout waiting for the write lock
#   "busy" / "busy snapshot" -- the engine's Busy / BusySnapshot errors on lock or snapshot contention
# Deliberately not a substring match on "conflict": ON CONFLICT clause errors and
# constraint failures are deterministic and must fail immediately.
_RETRYABLE_MESSAGES = (
    "write-write conflict",
    "database is locked",
    "database is busy",
    "busy snapshot",
    "transaction error: busy",
)


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


def connect_args_for_mode(
    config: TursoConfig | None = None, *, immediate_writes: bool = False
) -> dict[str, Any]:
    """Driver connect kwargs the journal mode requires.

    MVCC, and WAL engines with ``immediate_writes``, need driver autocommit so the
    ``begin`` listener controls the transaction statement; other WAL engines keep
    the driver's implicit ``BEGIN DEFERRED``. Pass the same ``immediate_writes`` to
    :func:`configure_engine`.
    """
    config = config or get_turso_config()
    explicit_begin = config.concurrent_writes or immediate_writes
    return {"isolation_level": None} if explicit_begin else {}


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


def install_transaction_hook(
    engine: AsyncEngine, config: TursoConfig | None = None, *, immediate_writes: bool = False
) -> None:
    """Have the engine emit its own ``BEGIN`` (see :func:`begin_statement`).

    In mvcc mode always; in wal mode only with ``immediate_writes``. Requires the
    engine's connections to be in driver autocommit (see :func:`connect_args_for_mode`).
    """
    config = config or get_turso_config()
    if not (config.concurrent_writes or immediate_writes):
        return

    @event.listens_for(engine.sync_engine, "begin")
    def _turso_begin(connection):
        statement = begin_statement(config, ddl=_exclusive_ddl.get(), write=_write_txn.get())
        connection.exec_driver_sql(statement or "BEGIN")


def configure_engine(
    engine: AsyncEngine,
    *,
    foreign_keys: bool = False,
    config: TursoConfig | None = None,
    immediate_writes: bool = False,
) -> None:
    """Install cognee's connection PRAGMAs and, when needed, the transaction hook.

    ``immediate_writes`` (graph engine) makes writes inside :func:`write_transaction`
    take the write lock up front in wal mode; build the engine with the matching
    :func:`connect_args_for_mode`.
    """
    config = config or get_turso_config()
    install_connect_pragmas(engine, connect_pragmas(config, foreign_keys=foreign_keys))
    install_transaction_hook(engine, config, immediate_writes=immediate_writes)


def begin_statement(
    config: TursoConfig | None = None, *, ddl: bool = False, write: bool = False
) -> str | None:
    """Statement that opens a transaction, or None to rely on the driver / autocommit.

    mvcc: ``BEGIN CONCURRENT``, or a plain exclusive ``BEGIN`` for DDL. wal:
    ``BEGIN IMMEDIATE`` for a write transaction that reads before it writes, so the
    read already holds the write lock; otherwise None.
    """
    config = config or get_turso_config()
    if config.concurrent_writes:
        return "BEGIN" if ddl else "BEGIN CONCURRENT"
    return "BEGIN IMMEDIATE" if write else None


@asynccontextmanager
async def exclusive_transaction() -> AsyncIterator[None]:
    """Run schema changes inside; MVCC engines then open plain ``BEGIN`` transactions."""
    token = _exclusive_ddl.set(True)
    try:
        yield
    finally:
        _exclusive_ddl.reset(token)


@asynccontextmanager
async def write_transaction() -> AsyncIterator[None]:
    """Run read-modify-write transactions inside; wal engines then use ``BEGIN IMMEDIATE``.

    Only engines configured with ``immediate_writes`` act on it; in mvcc mode the
    transaction stays ``BEGIN CONCURRENT``, whose conflicts the caller retries.
    """
    token = _write_txn.set(True)
    try:
        yield
    finally:
        _write_txn.reset(token)


def is_retryable_conflict(error: BaseException) -> bool:
    """True for the engine errors that a fresh attempt of the same transaction may clear.

    Matches the engine's exact contention messages. A SQLAlchemy ``DBAPIError``
    is unwrapped to the driver error first: its own ``str()`` appends the SQL and
    the bound parameters, so user text containing "database is locked" would make
    a constraint failure look retryable. Deterministic failures such as
    constraint violations or misconfigured transaction modes are never retried.
    """
    message = str(getattr(error, "orig", None) or error).lower()
    return any(known in message for known in _RETRYABLE_MESSAGES)


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
