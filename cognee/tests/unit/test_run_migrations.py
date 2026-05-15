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


if __name__ == "__main__":
    unittest.main()
