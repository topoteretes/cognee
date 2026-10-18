"""CLI status must reflect MCP startup, not merely the requested startup flag."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from cognee.cli._cognee import main


@pytest.mark.parametrize("mcp_started", [False, True])
def test_ui_summary_reports_whether_mcp_was_started(mcp_started):
    parser = MagicMock()
    parser.parse_args.return_value = argparse.Namespace(start_ui=True)
    frontend = MagicMock()
    frontend.poll.return_value = 0

    def start_ui(*, pid_callback, **kwargs):
        pid_callback(101)
        if mcp_started:
            pid_callback((102, "cognee-mcp-test"))
        return frontend

    with (
        patch("cognee.cli._cognee._create_parser", return_value=(parser, {})),
        patch("cognee.cli._cognee.signal.signal"),
        patch("cognee.cli._cognee.fmt") as fmt,
        patch("cognee.start_ui", side_effect=start_ui),
    ):
        assert main() == 0

    echoes = [call.args[0] for call in fmt.echo.call_args_list]
    notes = [call.args[0] for call in fmt.note.call_args_list]
    assert any("interface is available" in text for text in echoes)
    assert any("API backend is available" in text for text in echoes)
    assert any("MCP server is available" in text for text in echoes) is mcp_started
    assert any("MCP server was skipped" in text for text in notes) is not mcp_started
