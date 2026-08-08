"""Fresh-install error classification in get_default_user.

On a pristine install the SQLite databases directory does not exist yet, so the
first session raises OperationalError("unable to open database file") — or, if
the file exists but the schema does not, "no such table: ...". Both must be
classified as DatabaseNotCreatedError so the CLI's auto-migration recovery
(cognee/cli/user_resolution.py) can create the database and retry, instead of
surfacing a raw sqlite error to the user (COG-5973).
"""

import importlib
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError

gdu_mod = importlib.import_module("cognee.modules.users.methods.get_default_user")


def _engine_raising(error: Exception) -> MagicMock:
    """A relational-engine stub whose session context manager raises `error`."""

    @asynccontextmanager
    async def failing_session():
        raise error
        yield  # pragma: no cover

    engine = MagicMock()
    engine.get_async_session = failing_session
    return engine


def _sqlite_operational_error(message: str) -> OperationalError:
    import sqlite3

    return OperationalError("SELECT ...", {}, sqlite3.OperationalError(message))


class TestGetDefaultUserDatabaseNotCreated:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message",
        [
            "unable to open database file",
            "no such table: principals",
            "no such table: users",
        ],
    )
    async def test_fresh_install_errors_become_database_not_created(self, message):
        engine = _engine_raising(_sqlite_operational_error(message))
        with patch.object(gdu_mod, "get_relational_engine", return_value=engine):
            with pytest.raises(DatabaseNotCreatedError):
                await gdu_mod.get_default_user()

    @pytest.mark.asyncio
    async def test_unrelated_operational_error_propagates_unchanged(self):
        # "database is locked" is a live-database condition, not a missing one —
        # converting it would make the recovery path mask real contention issues.
        error = _sqlite_operational_error("database is locked")
        engine = _engine_raising(error)
        with patch.object(gdu_mod, "get_relational_engine", return_value=engine):
            with pytest.raises(OperationalError):
                await gdu_mod.get_default_user()

    @pytest.mark.asyncio
    async def test_non_operational_error_propagates_unchanged(self):
        engine = _engine_raising(RuntimeError("unable to open database file lookalike"))
        with patch.object(gdu_mod, "get_relational_engine", return_value=engine):
            with pytest.raises(RuntimeError):
                await gdu_mod.get_default_user()

    def test_database_not_created_error_does_not_self_log(self):
        # The exception base class logs itself on construction by default.
        # DatabaseNotCreatedError is a recoverable signal (the CLI catches it
        # and auto-creates the database), so it must stay silent — an ERROR
        # line on stderr during a successful fresh-install `cognee-cli add`
        # both alarms users and fails the CLI integration test.
        import cognee.exceptions.exceptions as exc_mod

        with patch.object(exc_mod, "logger") as mock_logger:
            DatabaseNotCreatedError()
        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_not_called()
