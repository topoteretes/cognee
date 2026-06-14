"""Tests for startup migrations (cognee/modules/migrations/startup.py + runner).

Originally written against cognee/run_migrations.py; the implementation moved
to cognee.modules.migrations.{startup,runner} (run_migrations.py is now a thin
shim). The original intents are preserved:

  * Alembic runs IN-PROCESS so programmatic cognee config is honored
  * a missing dataset_database table must skip migrations, not crash startup

plus the newer guarantees: empty-database bootstrap fallback and the
once-per-process flag being set only on clean runs.
"""

import asyncio
import importlib
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _reset_startup_flag():
    startup = importlib.import_module("cognee.modules.migrations.startup")
    startup._startup_migrations_done = False
    startup._startup_migrations_lock = None
    startup._startup_migrations_lock_loop = None


class TestRunRelationalMigrations(unittest.TestCase):
    """Alembic must run IN-PROCESS so programmatic cognee config is honored.

    A subprocess (the previous implementation) inherits only environment
    variables, so tests/examples that set roots via
    ``cognee.config.system_root_directory(...)`` had their migrations applied
    to the DEFAULT-location database instead — the library-test CI failure.
    (This also retires GitHub issue #2466's bare-'python' hazard: there is no
    subprocess interpreter to resolve anymore.)
    """

    def test_runs_alembic_in_process_without_logger_clobbering(self):
        shim = importlib.import_module("cognee.run_migrations")
        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.object(startup.pkg_resources, "files", return_value="/fake/package"),
            patch.object(startup.os.path, "exists", return_value=True),
            patch("alembic.command.upgrade") as mock_upgrade,
        ):
            asyncio.run(shim.run_migrations())

        mock_upgrade.assert_called_once()
        alembic_config, target = mock_upgrade.call_args[0]
        self.assertEqual(target, "head")
        # os.path.join -> backslash on Windows; assert platform-agnostically.
        self.assertEqual(
            alembic_config.get_main_option("script_location"),
            os.path.join("/fake/package", "alembic"),
        )
        # env.py honors this to skip fileConfig(), which would otherwise
        # disable the host process's loggers.
        self.assertFalse(alembic_config.attributes.get("configure_logger", True))


class TestMissingBookkeepingTables(unittest.TestCase):
    """run_database_migrations must skip gracefully when the bookkeeping tables
    are absent (fresh database, migrations invoked before any bootstrap)."""

    def _assert_skips(self, error):
        """Returns [] (no raise) when the first bookkeeping read raises ``error``."""
        runner = importlib.import_module("cognee.modules.migrations.runner")

        with (
            patch.object(runner, "backend_access_control_enabled", return_value=True),
            patch.object(runner, "_read_deployment_version", new=AsyncMock(return_value=None)),
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
            patch.object(runner, "_read_deployment_version", new=AsyncMock(return_value=None)),
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
        # Default to the "existing database -> upgrade" path so tests that only
        # care about the graph/vector flow don't hit a real DB inspection. The
        # fresh-vs-existing branch itself is covered by the two tests below.
        startup = importlib.import_module("cognee.modules.migrations.startup")
        self._schema_patch = patch.object(
            startup, "_relational_schema_exists", new=AsyncMock(return_value=True)
        )
        self._schema_patch.start()

    def tearDown(self):
        self._schema_patch.stop()
        _reset_startup_flag()

    def test_fresh_db_creates_schema_and_stamps_head(self):
        """An empty database (no users/alembic_version table) is created with
        create_all and STAMPED at head — never replaying historical migrations."""
        startup = importlib.import_module("cognee.modules.migrations.startup")

        relational = AsyncMock()
        stamp = AsyncMock()
        db_engine = MagicMock()
        db_engine.create_database = AsyncMock()

        with (
            patch.object(startup, "_relational_schema_exists", new=AsyncMock(return_value=False)),
            patch.object(startup, "run_relational_migrations", relational),
            patch.object(startup, "run_relational_stamp", stamp),
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

        db_engine.create_database.assert_awaited_once()
        stamp.assert_awaited_once_with("head")
        relational.assert_not_awaited()  # never run migrations on a fresh DB
        self.assertTrue(startup._startup_migrations_done)

    def test_existing_db_upgrades_and_does_not_stamp(self):
        """An existing database (users/alembic_version present) takes the upgrade
        path; a migration failure there is a real error that propagates — we do
        not create/stamp over it."""
        startup = importlib.import_module("cognee.modules.migrations.startup")

        stamp = AsyncMock()
        db_engine = MagicMock()
        db_engine.create_database = AsyncMock()

        with (
            patch.object(startup, "_relational_schema_exists", new=AsyncMock(return_value=True)),
            patch.object(
                startup,
                "run_relational_migrations",
                new=AsyncMock(side_effect=startup.MigrationError("boom")),
            ),
            patch.object(startup, "run_relational_stamp", stamp),
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                return_value=db_engine,
            ),
        ):
            with self.assertRaises(startup.MigrationError):
                asyncio.run(startup.run_startup_migrations())

        stamp.assert_not_awaited()
        db_engine.create_database.assert_not_awaited()
        self.assertFalse(startup._startup_migrations_done)

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

        # The very next call retries immediately and (now clean) sets the flag.
        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()) as relational,
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[{"dataset_id": "x", "migrations_applied": []}]),
            ),
        ):
            asyncio.run(startup.run_startup_migrations())
            relational.assert_awaited_once()
            self.assertTrue(startup._startup_migrations_done)

    def test_auto_migrations_flag_disables_all_runs(self):
        """ENABLE_AUTO_MIGRATIONS=false must skip relational AND data
        migrations entirely (read dynamically — tests set it after import).
        The explicit CLI path calls the runner directly and is unaffected."""
        import os

        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.dict(os.environ, {"ENABLE_AUTO_MIGRATIONS": "false"}),
            patch.object(startup, "run_relational_migrations", new=AsyncMock()) as relational,
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(),
            ) as database,
        ):
            asyncio.run(startup.run_startup_migrations())
            relational.assert_not_awaited()
            database.assert_not_awaited()
            self.assertFalse(startup._startup_migrations_done)

        # Flag back on -> next call runs normally (nothing was latched).
        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()) as relational,
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[]),
            ),
        ):
            asyncio.run(startup.run_startup_migrations())
            relational.assert_awaited_once()

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

    def test_returns_failed_databases_so_write_paths_can_block(self):
        """The failed-database ids are RETURNED (not just logged): write entry
        points (cognify/remember) treat a non-empty result as a hard stop so
        new-scheme writes can't land in an un-migrated store."""
        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()),
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(
                    return_value=[
                        {"dataset_id": "ds-1", "result": "failed"},
                        {"dataset_id": "ds-2", "migrations_applied": []},
                    ]
                ),
            ),
        ):
            failed = asyncio.run(startup.run_startup_migrations())
        self.assertEqual(failed, ["ds-1"])

    def test_returns_empty_list_when_all_succeed(self):
        startup = importlib.import_module("cognee.modules.migrations.startup")

        with (
            patch.object(startup, "run_relational_migrations", new=AsyncMock()),
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[{"dataset_id": "ds-1", "migrations_applied": []}]),
            ),
        ):
            failed = asyncio.run(startup.run_startup_migrations())
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
