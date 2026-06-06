"""Tests for cognee/run_migrations.py — ensures subprocess uses sys.executable."""

import sys
import unittest
import importlib
from unittest.mock import patch, MagicMock, AsyncMock


class TestRunMigrations(unittest.TestCase):
    """Verify run_migrations() invokes alembic via sys.executable, not bare 'python'."""

    def test_uses_sys_executable(self):
        """subprocess.run must be called with sys.executable, not 'python'.

        On Windows with uv-managed Python, bare 'python' can resolve to a
        different interpreter that doesn't have alembic installed
        (see GitHub issue #2466).
        """
        import asyncio

        module = importlib.import_module("cognee.run_migrations")

        with (
            patch.object(module.pkg_resources, "files", return_value="/fake/package"),
            patch.object(module.os.path, "exists", return_value=True),
            patch.object(module.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            asyncio.run(module.run_migrations())

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(
            cmd[0],
            sys.executable,
            f"Expected sys.executable ({sys.executable!r}) but got {cmd[0]!r}. "
            "Using bare 'python' breaks venv/uv setups on Windows.",
        )
        self.assertEqual(cmd[1:], ["-m", "alembic", "upgrade", "head"])


class TestRunVectorMigrations(unittest.TestCase):
    """run_vector_migrations() must skip gracefully when dataset_database is absent."""

    def _assert_skips(self, error):
        """run_vector_migrations returns [] (no raise) when the getter raises ``error``."""
        import asyncio

        module = importlib.import_module("cognee.run_migrations")
        with patch(
            "cognee.modules.data.methods.get_dataset_databases.get_dataset_databases",
            new=AsyncMock(side_effect=error),
        ):
            result = asyncio.run(module.run_vector_migrations())
        self.assertEqual(result, [])

    def test_skips_on_postgres_undefined_table(self):
        """PostgreSQL/asyncpg raises ProgrammingError for a missing table (issue: the
        dataset_database relation does not exist on a fresh DB). It must be caught
        like SQLite's OperationalError, not propagate and crash startup."""
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


if __name__ == "__main__":
    unittest.main()
