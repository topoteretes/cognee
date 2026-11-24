"""Recognize a Postgres deadlock abort behind SQLAlchemy's error wrapping."""

from sqlalchemy.exc import DBAPIError

DEADLOCK_SQLSTATE = "40P01"


def is_deadlock(error: BaseException) -> bool:
    """Whether Postgres aborted the statement to break a deadlock.

    SQLAlchemy raises its own ``DBAPIError`` and keeps the driver's error as
    ``orig``; with asyncpg that is an adapter error whose cause is the driver's
    ``DeadlockDetectedError``. The chain is walked for the deadlock sqlstate, so
    no driver is imported: the session cache runs on SQLite too.
    """
    if not isinstance(error, DBAPIError):
        return False
    cause = error.orig
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if getattr(cause, "sqlstate", None) == DEADLOCK_SQLSTATE:
            return True
        cause = cause.__cause__
    return False
