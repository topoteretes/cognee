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

    def test_run_vector_migrations_sweeps_dataset_database_rows(self):
        import asyncio
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )

        module = importlib.import_module("cognee.run_migrations")
        resolve_module = importlib.import_module(
            "cognee.infrastructure.databases.utils.resolve_dataset_database_connection_info"
        )

        dataset_rows = [
            MagicMock(
                dataset_id="dataset-1",
                vector_database_provider="lancedb",
                vector_database_url="/tmp/dataset-1.lance.db",
                vector_database_name="dataset-1.lance.db",
                vector_database_key="",
                vector_database_connection_info={},
                vector_dataset_database_handler="lancedb",
            ),
            MagicMock(
                dataset_id="dataset-2",
                vector_database_provider="pgvector",
                vector_database_url="postgres://ignored",
                vector_database_name="dataset-2",
                vector_database_key="",
                vector_database_connection_info={},
                vector_dataset_database_handler="pgvector",
            ),
        ]

        vector_engine = MagicMock()
        vector_engine.run_migrations = AsyncMock(return_value={"ok": True})
        # Defensive: if a leaked engine reaches await prune(), keep it awaitable.
        vector_engine.prune = AsyncMock(return_value=None)

        _create_vector_engine.cache_clear()
        try:
            with (
                patch(
                    "cognee.modules.data.methods.get_dataset_databases.get_dataset_databases",
                    new=AsyncMock(return_value=dataset_rows),
                ),
                patch.object(
                    resolve_module,
                    "resolve_dataset_database_connection_info",
                    new=AsyncMock(side_effect=lambda dataset_database: dataset_database),
                ),
                patch(
                    "cognee.infrastructure.databases.vector.create_vector_engine.create_vector_engine",
                    return_value=vector_engine,
                ) as mock_create_vector_engine,
            ):
                result = asyncio.run(module.run_vector_migrations())
        finally:
            _create_vector_engine.cache_clear()

        self.assertEqual(mock_create_vector_engine.call_count, 2)
        self.assertEqual(vector_engine.run_migrations.await_count, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["dataset_id"], "dataset-1")
        self.assertEqual(result[1]["dataset_id"], "dataset-2")


if __name__ == "__main__":
    unittest.main()
