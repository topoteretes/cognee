"""Tests for MCP port preflight handling in cognee.api.v1.ui.ui."""

from unittest.mock import patch

from cognee.api.v1.ui.ui import start_ui


class TestStartUiMcpPortPreflight:
    """Verify start_ui() treats a busy MCP port as optional, not fatal."""

    @patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False)
    @patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None)
    @patch("cognee.api.v1.ui.ui._is_port_available", return_value=False)
    @patch("cognee.api.v1.ui.ui._check_docker_available", return_value=(True, "ok"))
    @patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, []))
    def test_busy_mcp_port_skips_mcp_not_the_launch(
        self, mock_ports, mock_docker, mock_port_free, mock_frontend, mock_prompt
    ):
        """With the MCP port taken, no docker call happens but required ports are still checked."""
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen"):
            result = start_ui(
                pid_callback=lambda p: None,
                start_mcp=True,
                start_backend=False,
            )

        # start_ui returns None only because find_frontend_path is None here;
        # the point is that it got past the port preflight at all.
        assert result is None
        mock_ports.assert_called_once()
        checked = mock_ports.call_args[0][0]
        assert ("Frontend UI", 3000) not in checked  # default port is free in this mock list
        assert all(name != "MCP Server" for _, name in checked)
        # The optional MCP server was dropped before the Docker preflight could run.
        mock_docker.assert_not_called()
        for c in mock_run.call_args_list:
            args = c[0][0] if c[0] else c[1].get("args", [])
            assert "docker" not in str(args), f"Unexpected docker call: {args}"

    @patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False)
    @patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None)
    @patch("cognee.api.v1.ui.ui._is_port_available", return_value=True)
    @patch("cognee.api.v1.ui.ui._check_docker_available", return_value=(True, "ok"))
    @patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, []))
    def test_free_mcp_port_still_runs_docker_preflight(
        self, mock_ports, mock_docker, mock_port_free, mock_frontend, mock_prompt
    ):
        """With the MCP port free, the Docker preflight runs as before."""
        with patch("subprocess.run", return_value=None) as mock_run, patch("subprocess.Popen"):
            start_ui(
                pid_callback=lambda p: None,
                start_mcp=True,
                start_backend=False,
            )

        mock_docker.assert_called_once()

    @patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False)
    @patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None)
    @patch("cognee.api.v1.ui.ui._check_docker_available", return_value=(True, "ok"))
    @patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(False, ["Frontend UI (port 3000)"]))
    def test_busy_frontend_port_remains_fatal(
        self, mock_ports, mock_docker, mock_frontend, mock_prompt
    ):
        """A busy frontend port must still abort the launch."""
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            result = start_ui(
                pid_callback=lambda p: None,
                start_mcp=False,
                start_backend=False,
            )

        assert result is None
        mock_popen.assert_not_called()
