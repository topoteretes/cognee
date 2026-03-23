"""Tests for cognee/run_migrations.py — ensures subprocess uses sys.executable."""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestRunMigrations(unittest.TestCase):
    """Verify run_migrations() invokes alembic via sys.executable, not bare 'python'."""

    @patch("cognee.run_migrations.subprocess.run")
    @patch("cognee.run_migrations.os.path.exists", return_value=True)
    @patch("cognee.run_migrations.pkg_resources.files", return_value="/fake/package")
    def test_uses_sys_executable(self, mock_files, mock_exists, mock_run):
        """subprocess.run must be called with sys.executable, not 'python'.

        On Windows with uv-managed Python, bare 'python' can resolve to a
        different interpreter that doesn't have alembic installed
        (see GitHub issue #2466).
        """
        import asyncio
        from cognee.run_migrations import run_migrations

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        asyncio.run(run_migrations())

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
