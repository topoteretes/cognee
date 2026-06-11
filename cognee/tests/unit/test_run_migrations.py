"""Tests for startup migrations (cognee/modules/migrations/startup.py + runner).

Originally written against cognee/run_migrations.py; the implementation moved
to cognee.modules.migrations.{startup,runner} (run_migrations.py is now a thin
shim). The original intents are preserved:

  * the Alembic subprocess must use sys.executable, not bare 'python'
  * a missing dataset_database table must skip migrations, not crash startup

plus the newer guarantees: empty-database bootstrap fallback and the
once-per-process flag being set only on clean runs.
"""

import asyncio
import importlib
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _reset_startup_flag():
    startup = importlib.import_module("cognee.modules.migrations.startup")
    startup._startup_migrations_done = False
    startup._startup_migrations_lock = None
    startup._startup_migrations_lock_loop = None


class TestRunRelationalMigrations(unittest.TestCase):
    """Verify the Alembic subprocess is invoked via sys.executable, not bare 'python'."""

    def test_uses_sys_executable(self):
        """subprocess.run must be called with sys.executable, not 'python'.

        On Windows with uv-managed Python, bare 'python' can resolve to a
        different interpreter that doesn't have alembic installed
        (see GitHub issue #2466). Exercised through the public shim
        (cognee.run_migrations.run_migrations) so the delegation is covered too.
        """
        shim = importlib.import_module("cognee.run_migrations")
        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.object(startup.pkg_resources, "files", return_value="/fake/package"),
            patch.object(startup.os.path, "exists", return_value=True),
            patch.object(startup.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            asyncio.run(shim.run_migrations())

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(
            cmd[0],
            sys.executable,
            f"Expected sys.executable ({sys.executable!r}) but got {cmd[0]!r}. "
            "Using bare 'python' breaks venv/uv setups on Windows.",
        )
        self.assertEqual(cmd[1:], ["-m", "alembic", "upgrade", "head"])


class TestMissingBookkeepingTables(unittest.TestCase):
    """run_database_migrations must skip gracefully when the bookkeeping tables
    are absent (fresh database, migrations invoked before any bootstrap)."""

    def _assert_skips(self, error):
        """Returns [] (no raise) when the first bookkeeping read raises ``error``."""
        runner = importlib.import_module("cognee.modules.migrations.runner")

        with (
            patch.object(runner, "backend_access_control_enabled", return_value=True),
            patch.object(runner, "_record_deployment_version", new=AsyncMock(side_effect=error)),
        ):
            result = asyncio.run(runner.run_database_migrations())
        self.assertEqual(result, [])

    def test_skips_on_postgres_undefined_table(self):
        """PostgreSQL/asyncpg raises ProgrammingError for a missing table (the
        dataset_database relation does not exist on a fresh DB). It must be
        caught like SQLite's OperationalError, not propagate and crash startup."""
        from sqlalchemy.exc import ProgrammingError

        self._assert_skips(
            ProgrammingError(
                "SELECT * FROM dataset_database",
                {},
                Exception('relation "dataset_database" does not exist'),
            )
        )

    def test_skips_on_sqlite_missing_table(self):
        """SQLite raises OperationalError ('no such table') — the original guarded case."""
        from sqlalchemy.exc import OperationalError

        self._assert_skips(
            OperationalError("SELECT * FROM dataset_database", {}, Exception("no such table"))
        )

    def test_skips_when_dataset_rows_unreadable(self):
        """Same tolerance when get_dataset_databases itself raises (original test shape)."""
        from sqlalchemy.exc import OperationalError

        runner = importlib.import_module("cognee.modules.migrations.runner")

        with (
            patch.object(runner, "backend_access_control_enabled", return_value=True),
            patch.object(runner, "_record_deployment_version", new=AsyncMock(return_value=None)),
            patch.object(
                runner,
                "get_dataset_databases",
                new=AsyncMock(
                    side_effect=OperationalError("SELECT", {}, Exception("no such table"))
                ),
            ),
        ):
            result = asyncio.run(runner.run_database_migrations())
        self.assertEqual(result, [])


class TestStartupMigrationsBootstrap(unittest.TestCase):
    """run_startup_migrations: empty-DB bootstrap fallback + once-per-process flag."""

    def setUp(self):
        _reset_startup_flag()

    def tearDown(self):
        _reset_startup_flag()

    def test_bootstraps_schema_and_retries_when_alembic_fails_on_empty_db(self):
        """cognee's Alembic chain is not self-sufficient on an empty database;
        the first failure must trigger create_database() and ONE retry — the
        recovery remember()/MCP/SDK callers rely on (the API lifespan has its
        own copy)."""
        startup = importlib.import_module("cognee.modules.migrations.startup")

        relational = AsyncMock(side_effect=[startup.MigrationError("empty db"), None])
        db_engine = MagicMock()
        db_engine.create_database = AsyncMock()

        with (
            patch.object(startup, "run_relational_migrations", relational),
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                return_value=db_engine,
            ),
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[]),
            ),
        ):
            asyncio.run(startup.run_startup_migrations())

        self.assertEqual(relational.await_count, 2)
        db_engine.create_database.assert_awaited_once()
        self.assertTrue(startup._startup_migrations_done)

    def test_flag_not_set_when_a_database_failed(self):
        """A failed dataset must be retried by the next call in this process —
        the flag is set only on clean runs."""
        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()),
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[{"dataset_id": "x", "result": "failed"}]),
            ),
        ):
            asyncio.run(startup.run_startup_migrations())
            self.assertFalse(startup._startup_migrations_done)

        # Next call (now clean) runs again and sets the flag.
        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()) as relational,
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[{"dataset_id": "x", "graph_migrations_applied": []}]),
            ),
        ):
            asyncio.run(startup.run_startup_migrations())
            relational.assert_awaited_once()
            self.assertTrue(startup._startup_migrations_done)

    def test_second_call_is_a_noop_after_success(self):
        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()) as relational,
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[]),
            ),
        ):
            asyncio.run(startup.run_startup_migrations())
            asyncio.run(startup.run_startup_migrations())
            relational.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
