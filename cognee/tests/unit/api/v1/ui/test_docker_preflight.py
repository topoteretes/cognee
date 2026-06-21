"""Tests for Docker daemon preflight checks in cognee.api.v1.ui.ui."""

import subprocess
from unittest.mock import patch, MagicMock, call

from cognee.api.v1.ui.ui import _check_docker_available


class TestCheckDockerAvailable:
    """Unit tests for _check_docker_available()."""

    def test_docker_not_on_path(self):
        """When the docker binary is not on PATH, return False with install guidance."""
        with patch("shutil.which", return_value=None):
            ok, msg = _check_docker_available()

        assert ok is False
        assert "not found on PATH" in msg
        assert "Colima" in msg

    def test_docker_daemon_running(self):
        """When `docker info` succeeds, return True."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", return_value=mock_result),
        ):
            ok, msg = _check_docker_available()

        assert ok is True
        assert "running" in msg.lower()

    def test_docker_daemon_not_running(self):
        """When `docker info` fails (rc != 0), return False with actionable guidance."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"Cannot connect to the Docker daemon"

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", return_value=mock_result),
        ):
            ok, msg = _check_docker_available()

        assert ok is False
        assert "not responding" in msg
        assert "Colima" in msg
        assert "colima start" in msg
        # Verify the actual stderr from `docker info` is passed through
        assert "Cannot connect to the Docker daemon" in msg

    def test_docker_info_timeout(self):
        """When `docker info` times out, return False with retry guidance."""
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="docker info", timeout=15),
            ),
        ):
            ok, msg = _check_docker_available()

        assert ok is False
        assert "15 seconds" in msg

    def test_docker_binary_missing_at_runtime(self):
        """When shutil.which finds docker but subprocess.run raises FileNotFoundError."""
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            ok, msg = _check_docker_available()

        assert ok is False
        # FileNotFoundError is handled by the broadened (OSError, SubprocessError)
        # branch; the message names the error type and still points at install docs.
        assert "FileNotFoundError" in msg
        assert "Colima" in msg

    def test_docker_exec_permission_error(self):
        """A PermissionError at exec time (OSError subclass) must be caught, not raised."""
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", side_effect=PermissionError("[Errno 13] Permission denied")),
        ):
            ok, msg = _check_docker_available()

        assert ok is False
        assert "PermissionError" in msg
        assert "could not be executed" in msg.lower()


class TestStartUiDockerIntegration:
    """Verify start_ui() gracefully degrades when Docker is unavailable."""

    @patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False)
    @patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None)
    @patch("cognee.api.v1.ui.ui._check_docker_available", return_value=(False, "no docker"))
    @patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, []))
    def test_skips_mcp_docker_pull_when_docker_unavailable(
        self, mock_ports, mock_docker, mock_frontend, mock_prompt
    ):
        """When Docker is unavailable, start_ui must not call docker pull/run."""
        from cognee.api.v1.ui.ui import start_ui

        pids = []
        # start_ui will return None because there's no frontend, but the key
        # assertion is that it never attempts docker pull (subprocess.run with
        # ['docker', 'pull', ...]).
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            start_ui(
                pid_callback=lambda p: pids.append(p),
                start_mcp=True,
                start_backend=False,
            )

            # docker pull / docker run should never have been called
            for c in mock_run.call_args_list:
                args = c[0][0] if c[0] else c[1].get("args", [])
                assert "docker" not in str(args), f"Unexpected docker call: {args}"

            mock_popen.assert_not_called()

    @patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False)
    @patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None)
    @patch("cognee.api.v1.ui.ui._check_docker_available", return_value=(False, "no docker"))
    @patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, []))
    def test_docker_check_not_called_when_mcp_disabled(
        self, mock_ports, mock_docker, mock_frontend, mock_prompt
    ):
        """When start_mcp=False, _check_docker_available should not be called."""
        from cognee.api.v1.ui.ui import start_ui

        start_ui(
            pid_callback=lambda p: None,
            start_mcp=False,
            start_backend=False,
        )

        mock_docker.assert_not_called()
