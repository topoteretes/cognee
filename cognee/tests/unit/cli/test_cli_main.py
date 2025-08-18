"""
Tests for the main CLI entry point and command discovery.
"""

import pytest
import argparse
from unittest.mock import patch, MagicMock
from cognee.cli._cognee import main, _discover_commands, _create_parser
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class TestCliMain:
    """Test the main CLI functionality"""

    def test_discover_commands(self):
        """Test that all expected commands are discovered"""
        commands = _discover_commands()

        # Check that we get command classes back
        assert len(commands) > 0

        # Check that we have the expected commands
        command_strings = []
        for command_class in commands:
            command = command_class()
            command_strings.append(command.command_string)

        expected_commands = ["add", "search", "cognify", "delete", "config"]
        for expected_command in expected_commands:
            assert expected_command in command_strings

    def test_create_parser(self):
        """Test parser creation and command installation"""
        parser, installed_commands = _create_parser()

        # Check parser is created
        assert isinstance(parser, argparse.ArgumentParser)

        # Check commands are installed
        expected_commands = ["add", "search", "cognify", "delete", "config"]
        for expected_command in expected_commands:
            assert expected_command in installed_commands

        # Check parser has version argument
        actions = [action.dest for action in parser._actions]
        assert "version" in actions

    @patch("cognee.cli._cognee._create_parser")
    def test_main_no_command(self, mock_create_parser):
        """Test main function when no command is provided"""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = MagicMock(command=None)
        mock_create_parser.return_value = (mock_parser, {})

        result = main()

        assert result == -1
        mock_parser.print_help.assert_called_once()

    @patch("cognee.cli._cognee._create_parser")
    def test_main_with_valid_command(self, mock_create_parser):
        """Test main function with a valid command"""
        mock_command = MagicMock()
        mock_command.execute.return_value = None

        mock_parser = MagicMock()
        mock_args = MagicMock(command="test")
        mock_parser.parse_args.return_value = mock_args

        mock_create_parser.return_value = (mock_parser, {"test": mock_command})

        result = main()

        assert result == 0
        mock_command.execute.assert_called_once_with(mock_args)

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.debug.is_debug_enabled")
    def test_main_with_command_exception(self, mock_debug, mock_create_parser):
        """Test main function when command raises exception"""
        mock_debug.return_value = False

        mock_command = MagicMock()
        mock_command.execute.side_effect = CliCommandException("Test error", error_code=2)

        mock_parser = MagicMock()
        mock_args = MagicMock(command="test")
        mock_parser.parse_args.return_value = mock_args

        mock_create_parser.return_value = (mock_parser, {"test": mock_command})

        result = main()

        assert result == 2

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.debug.is_debug_enabled")
    def test_main_with_generic_exception(self, mock_debug, mock_create_parser):
        """Test main function when command raises generic exception"""
        mock_debug.return_value = False

        mock_command = MagicMock()
        mock_command.execute.side_effect = Exception("Generic error")

        mock_parser = MagicMock()
        mock_args = MagicMock(command="test")
        mock_parser.parse_args.return_value = mock_args

        mock_create_parser.return_value = (mock_parser, {"test": mock_command})

        result = main()

        assert result == -1

    @patch("cognee.cli._cognee._create_parser")
    @patch("cognee.cli.debug.is_debug_enabled")
    def test_main_debug_mode_reraises_exception(self, mock_debug, mock_create_parser):
        """Test main function reraises exceptions in debug mode"""
        mock_debug.return_value = True

        test_exception = CliCommandException(
            "Test error", error_code=2, raiseable_exception=ValueError("Inner error")
        )

        mock_command = MagicMock()
        mock_command.execute.side_effect = test_exception

        mock_parser = MagicMock()
        mock_args = MagicMock(command="test")
        mock_parser.parse_args.return_value = mock_args

        mock_create_parser.return_value = (mock_parser, {"test": mock_command})

        with pytest.raises(ValueError, match="Inner error"):
            main()

    def test_version_argument(self):
        """Test that version argument is properly configured"""
        parser, _ = _create_parser()

        # Check that version action exists
        version_actions = [action for action in parser._actions if action.dest == "version"]
        assert len(version_actions) == 1

        version_action = version_actions[0]
        assert "cognee" in version_action.version

    def test_debug_argument(self):
        """Test that debug argument is properly configured"""
        parser, _ = _create_parser()

        # Check that debug action exists
        debug_actions = [action for action in parser._actions if action.dest == "debug"]
        assert len(debug_actions) == 1


class TestDebugAction:
    """Test the DebugAction class"""

    @patch("cognee.cli.debug.enable_debug")
    @patch("cognee.cli.echo.note")
    def test_debug_action_call(self, mock_note, mock_enable_debug):
        """Test that DebugAction enables debug mode"""
        from cognee.cli._cognee import DebugAction

        action = DebugAction([])
        parser = MagicMock()
        namespace = MagicMock()

        action(parser, namespace, None)

        mock_enable_debug.assert_called_once()
        mock_note.assert_called_once_with("Debug mode enabled. Full stack traces will be shown.")
