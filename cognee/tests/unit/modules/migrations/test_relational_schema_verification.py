"""The relational schema must actually match the models it is stamped against.

``apply_all_migrations`` either upgrades an existing database or, for a fresh
one, runs ``create_all`` and STAMPS Alembic head. Both paths only *assert* the
schema is at the target — the stamp records a revision without running it, and
the upgrade assumes every revision did what it claimed. When a deployment ships
an Alembic tree ahead of its bundled ORM models, a tenant is provisioned with a
schema that disagrees with its stamp while every signal (exit code,
``alembic current``, the logs) still reads healthy.

These tests reflect a REAL SQLite database rather than mocking the inspector, so
a change that stops detecting drift fails here.
"""

import asyncio
import importlib
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import Column, Integer, MetaData, String, Table
from sqlalchemy.ext.asyncio import create_async_engine

startup = importlib.import_module("cognee.modules.migrations.startup")


@asynccontextmanager
async def _null_async_cm(*args, **kwargs):
    yield


def _metadata(with_extra_column: bool, with_second_table: bool) -> MetaData:
    """Declare what the models expect: one table (optionally with a column the
    database will not have) plus an optional second table."""
    metadata = MetaData()
    columns = [Column("id", Integer, primary_key=True), Column("name", String)]
    if with_extra_column:
        columns.append(Column("user_id", String))
    Table("widgets", metadata, *columns)
    if with_second_table:
        Table("gadgets", metadata, Column("id", Integer, primary_key=True))
    return metadata


class _EngineHolder:
    """Stands in for the relational engine wrapper (`.engine` is the AsyncEngine)."""

    def __init__(self, engine):
        self.engine = engine


class TestSchemaDriftDetection(unittest.IsolatedAsyncioTestCase):
    """`_collect_relational_schema_drift` against a real database."""

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self._tmp.close()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self._tmp.name}")
        # The database only ever gets the two-column `widgets` table: this is the
        # tenant provisioned by an older model set.
        on_disk = MetaData()
        Table(
            "widgets",
            on_disk,
            Column("id", Integer, primary_key=True),
            Column("name", String),
        )
        async with self.engine.begin() as connection:
            await connection.run_sync(on_disk.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        os.unlink(self._tmp.name)

    def _patch_for(self, metadata):
        base = MagicMock()
        base.metadata = metadata
        return (
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                return_value=_EngineHolder(self.engine),
            ),
            patch(
                "cognee.infrastructure.databases.relational.ModelBase.Base",
                base,
            ),
        )

    async def test_reports_a_column_the_database_lacks(self):
        """The August failure: the model gained `user_id`, the tenant's table did not."""
        engine_patch, base_patch = self._patch_for(_metadata(True, False))
        with engine_patch, base_patch:
            drift = await startup._collect_relational_schema_drift()

        self.assertEqual(len(drift), 1, drift)
        self.assertIn("widgets", drift[0])
        self.assertIn("user_id", drift[0])

    async def test_reports_a_table_the_database_lacks(self):
        engine_patch, base_patch = self._patch_for(_metadata(False, True))
        with engine_patch, base_patch:
            drift = await startup._collect_relational_schema_drift()

        self.assertEqual(drift, ["missing table 'gadgets'"])

    async def test_matching_schema_reports_nothing(self):
        engine_patch, base_patch = self._patch_for(_metadata(False, False))
        with engine_patch, base_patch:
            drift = await startup._collect_relational_schema_drift()

        self.assertEqual(drift, [])

    async def test_extra_database_columns_are_not_drift(self):
        """Alembic bookkeeping and retired columns live in the database without
        being declared; only what the models NEED is checked."""
        async with self.engine.begin() as connection:
            extra = MetaData()
            Table("leftovers", extra, Column("id", Integer, primary_key=True))
            await connection.run_sync(extra.create_all)

        engine_patch, base_patch = self._patch_for(_metadata(False, False))
        with engine_patch, base_patch:
            drift = await startup._collect_relational_schema_drift()

        self.assertEqual(drift, [])

    async def test_uninspectable_database_is_not_a_failure(self):
        """The check is a safety net; it must never become a new way to fail."""
        broken = MagicMock()
        broken.engine.connect.side_effect = RuntimeError("no route to host")
        with (
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                return_value=broken,
            ),
            patch(
                "cognee.infrastructure.databases.relational.ModelBase.Base",
                MagicMock(metadata=_metadata(True, True)),
            ),
        ):
            drift = await startup._collect_relational_schema_drift()

        self.assertEqual(drift, [])


class TestSchemaCheckModes(unittest.IsolatedAsyncioTestCase):
    """`verify_relational_schema` honours MIGRATION_SCHEMA_CHECK."""

    def setUp(self):
        self._saved = os.environ.get("MIGRATION_SCHEMA_CHECK")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("MIGRATION_SCHEMA_CHECK", None)
        else:
            os.environ["MIGRATION_SCHEMA_CHECK"] = self._saved

    async def test_strict_raises_with_the_drift_in_the_message(self):
        os.environ["MIGRATION_SCHEMA_CHECK"] = "strict"
        with patch.object(
            startup,
            "_collect_relational_schema_drift",
            new=AsyncMock(return_value=["table 'pipeline_runs' is missing column(s): user_id"]),
        ):
            with self.assertRaises(startup.MigrationError) as caught:
                await startup.verify_relational_schema()

        self.assertIn("pipeline_runs", str(caught.exception))
        self.assertIn("user_id", str(caught.exception))

    async def test_warn_returns_drift_without_raising(self):
        os.environ["MIGRATION_SCHEMA_CHECK"] = "warn"
        with patch.object(
            startup,
            "_collect_relational_schema_drift",
            new=AsyncMock(return_value=["missing table 'gadgets'"]),
        ):
            drift = await startup.verify_relational_schema()

        self.assertEqual(drift, ["missing table 'gadgets'"])

    async def test_off_skips_the_comparison_entirely(self):
        os.environ["MIGRATION_SCHEMA_CHECK"] = "off"
        collect = AsyncMock(return_value=["missing table 'gadgets'"])
        with patch.object(startup, "_collect_relational_schema_drift", new=collect):
            drift = await startup.verify_relational_schema()

        self.assertEqual(drift, [])
        collect.assert_not_awaited()

    async def test_unknown_mode_falls_back_to_warn(self):
        os.environ["MIGRATION_SCHEMA_CHECK"] = "nonsense"
        with patch.object(
            startup,
            "_collect_relational_schema_drift",
            new=AsyncMock(return_value=["missing table 'gadgets'"]),
        ):
            drift = await startup.verify_relational_schema()

        self.assertEqual(drift, ["missing table 'gadgets'"])

    async def test_default_mode_is_warn(self):
        os.environ.pop("MIGRATION_SCHEMA_CHECK", None)
        self.assertEqual(startup._schema_check_mode(), "warn")


class TestVerificationRunsOnBothMigrationPaths(unittest.TestCase):
    """The check must run after the fresh-database stamp AND after an upgrade —
    a stamp is exactly where the assertion goes unverified."""

    def setUp(self):
        runner = importlib.import_module("cognee.modules.migrations.runner")
        self._lock_patch = patch.object(runner, "_migration_lock", _null_async_cm)
        self._lock_patch.start()

    def tearDown(self):
        self._lock_patch.stop()

    def _run(self, schema_exists):
        verify = AsyncMock(return_value=[])
        db_engine = MagicMock()
        db_engine.create_database = AsyncMock()
        with (
            patch.object(
                startup, "_relational_schema_exists", new=AsyncMock(return_value=schema_exists)
            ),
            patch.object(startup, "run_relational_migrations", new=AsyncMock()),
            patch.object(startup, "run_relational_stamp", new=AsyncMock()),
            patch.object(startup, "verify_relational_schema", verify),
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                return_value=db_engine,
            ),
            patch(
                "cognee.modules.migrations.runner.run_database_migrations",
                new=AsyncMock(return_value=[]),
            ),
        ):
            asyncio.run(startup.apply_all_migrations("head"))
        return verify

    def test_runs_after_fresh_database_stamp(self):
        self._run(schema_exists=False).assert_awaited_once()

    def test_runs_after_existing_database_upgrade(self):
        self._run(schema_exists=True).assert_awaited_once()

    def test_strict_failure_stops_the_data_migrations(self):
        """Drift means the tables the graph/vector chain reads are wrong — it must
        not proceed on a schema we know is broken."""
        data_migrations = AsyncMock(return_value=[])
        db_engine = MagicMock()
        db_engine.create_database = AsyncMock()
        with (
            patch.object(startup, "_relational_schema_exists", new=AsyncMock(return_value=False)),
            patch.object(startup, "run_relational_stamp", new=AsyncMock()),
            patch.object(
                startup,
                "verify_relational_schema",
                new=AsyncMock(side_effect=startup.MigrationError("drift")),
            ),
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                return_value=db_engine,
            ),
            patch("cognee.modules.migrations.runner.run_database_migrations", new=data_migrations),
        ):
            with self.assertRaises(startup.MigrationError):
                asyncio.run(startup.apply_all_migrations("head"))

        data_migrations.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
