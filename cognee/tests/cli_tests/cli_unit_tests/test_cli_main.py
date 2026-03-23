"""Tests for the main CLI entry point, JSON mode, one-shot prompt, and interactive REPL."""

import json
import pytest
import argparse
from unittest.mock import patch, MagicMock
from cognee.cli._cognee import main, _discover_commands, _create_parser
from cognee.cli.exceptions import CliCommandException


def _base_args(**overrides):
    """Create a base argparse.Namespace with all required fields for main()."""
    defaults = dict(command=None, prompt=None, interactive=False, continue_session=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliCore:
    """Core CLI: discovery, command dispatch, error handling"""

    def test_discover_and_create_parser(self):
        commands = _discover_commands()
        assert len(commands) > 0

        parser, installed = _create_parser()
        for name in ["add", "search", "cognify", "delete", "config", "status", "datasets"]:
            assert name in installed

    @patch("cognee.cli._cognee._create_parser")
    def test_command_success(self, mock_create_parser):
        cmd = MagicMock()
        cmd.execute.return_value = {"message": "ok"}
        mock_parser = MagicMock()
        mock_args = _base_args(command="test", json_mode=True)
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = (mock_parser, {"test": cmd})

        assert main() == 0
        cmd.execute.assert_called_once_with(mock_args)

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.debug.is_debug_enabled", return_value=False)
    def test_command_exception_returns_error_code(self, _debug, mock_create_parser):
        cmd = MagicMock()
        cmd.execute.side_effect = CliCommandException("fail", error_code=2)
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(command="test", json_mode=True)
        mock_create_parser.return_value = (mock_parser, {"test": cmd})

        assert main() == 2


class TestJsonMode:
    """JSON envelope: success, error, interactive blocking"""

    def setup_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    def teardown_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    @patch("cognee.cli._cognee._create_parser")
    def test_json_success_envelope(self, mock_create_parser, capsys):
        cmd = MagicMock()
        cmd.execute.return_value = {"message": "Done."}
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(command="test", json_mode=True)
        mock_create_parser.return_value = (mock_parser, {"test": cmd})

        assert main() == 0
        output = json.loads(capsys.readouterr().out)
        assert output == {"status": "ok", "message": "Done."}

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.debug.is_debug_enabled", return_value=False)
    def test_json_error_envelope(self, _debug, mock_create_parser, capsys):
        cmd = MagicMock()
        cmd.execute.side_effect = CliCommandException("broke", error_code=3)
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(command="test", json_mode=True)
        mock_create_parser.return_value = (mock_parser, {"test": cmd})

        assert main() == 3
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "error"
        assert output["error_code"] == 3

    def test_confirm_blocked_in_json_mode(self):
        import cognee.cli.echo as fmt

        fmt.enable_json_mode()
        with pytest.raises(CliCommandException, match="--force with --json"):
            fmt.confirm("sure?")


class TestOneShotPrompt:
    """The -p flag: one-shot query"""

    def setup_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    def teardown_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.repl.run_prompt")
    @patch("cognee.cli.session.load_session")
    def test_prompt_returns_json(self, mock_session, mock_prompt, mock_create_parser, capsys):
        import cognee.cli.echo as fmt

        fmt.enable_json_mode()
        mock_session.return_value = {"dataset": "ds", "query_type": "GRAPH_COMPLETION"}
        mock_prompt.return_value = {"results": ["a"], "count": 1}
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(json_mode=True, prompt="question")
        mock_create_parser.return_value = (mock_parser, {})

        assert main() == 0
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "ok"
        assert output["count"] == 1


class TestInteractiveAndSession:
    """The -i, -c flags and session persistence"""

    def setup_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    def teardown_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.repl.run_interactive", return_value=0)
    def test_interactive_enters_repl(self, mock_repl, mock_create_parser):
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(interactive=True)
        mock_create_parser.return_value = (mock_parser, {})

        assert main() == 0
        mock_repl.assert_called_once_with({})

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.repl.run_interactive", return_value=0)
    @patch("cognee.cli.session.load_session")
    def test_continue_resumes_session(self, mock_session, mock_repl, mock_create_parser):
        mock_session.return_value = {"dataset": "saved", "query_type": "CHUNKS"}
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(continue_session=True)
        mock_create_parser.return_value = (mock_parser, {})

        assert main() == 0
        mock_repl.assert_called_once_with({"dataset": "saved", "query_type": "CHUNKS"})

    def test_session_round_trip(self, tmp_path):
        from cognee.cli import session

        orig_dir, orig_file = session.SESSION_DIR, session.SESSION_FILE
        session.SESSION_DIR = tmp_path
        session.SESSION_FILE = tmp_path / "session.json"
        try:
            session.save_session(dataset="my_data", query_type="CHUNKS")
            loaded = session.load_session()
            assert loaded["dataset"] == "my_data"
            assert loaded["query_type"] == "CHUNKS"
        finally:
            session.SESSION_DIR, session.SESSION_FILE = orig_dir, orig_file

    @patch("cognee.cli.repl._run_query", return_value=["r1", "r2"])
    @patch("cognee.cli.repl.save_session")
    @patch("builtins.input", side_effect=["test query", "/quit"])
    def test_repl_executes_query(self, _input, _save, mock_query):
        from cognee.cli.repl import run_interactive

        assert run_interactive({"dataset": "ds", "query_type": "CHUNKS"}) == 0
        mock_query.assert_called_once_with("test query", "CHUNKS", "ds")


class TestStatusCommand:
    """The status preflight check"""

    def setup_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    def teardown_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    @patch("cognee.cli._cognee._create_parser")
    def test_status_returns_ready_flag(self, mock_create_parser, capsys):
        cmd = MagicMock()
        cmd.execute.return_value = {"ready": True, "llm": {"api_key_set": True}}
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(command="status", json_mode=True)
        mock_create_parser.return_value = (mock_parser, {"status": cmd})

        assert main() == 0
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "ok"
        assert output["ready"] is True


class TestDatasetsCommand:
    """The datasets list command"""

    def setup_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    def teardown_method(self):
        import cognee.cli.echo as fmt

        fmt._JSON_MODE = False

    @patch("cognee.cli._cognee._create_parser")
    def test_datasets_list_returns_count(self, mock_create_parser, capsys):
        cmd = MagicMock()
        cmd.execute.return_value = {"datasets": [{"name": "test_ds"}], "count": 1}
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _base_args(command="datasets", json_mode=True)
        mock_create_parser.return_value = (mock_parser, {"datasets": cmd})

        assert main() == 0
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "ok"
        assert output["count"] == 1
