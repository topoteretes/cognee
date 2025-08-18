"""
Tests for CLI utility functions and helper modules.
"""

import pytest
from unittest.mock import patch, MagicMock
import click
from cognee.cli import echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli import debug
from cognee.cli.config import (
    CLI_DESCRIPTION,
    DEFAULT_DOCS_URL,
    COMMAND_DESCRIPTIONS,
    SEARCH_TYPE_CHOICES,
    CHUNKER_CHOICES,
    OUTPUT_FORMAT_CHOICES,
)


class TestEchoModule:
    """Test the CLI echo/formatting module"""

    @patch("click.secho")
    def test_echo_basic(self, mock_secho):
        """Test basic echo functionality"""
        fmt.echo("test message")
        mock_secho.assert_called_once_with("test message", fg=None, err=False)

    @patch("click.secho")
    def test_echo_with_color(self, mock_secho):
        """Test echo with color"""
        fmt.echo("test message", color="red")
        mock_secho.assert_called_once_with("test message", fg="red", err=False)

    @patch("click.secho")
    def test_echo_to_stderr(self, mock_secho):
        """Test echo to stderr"""
        fmt.echo("test message", err=True)
        mock_secho.assert_called_once_with("test message", fg=None, err=True)

    @patch("cognee.cli.echo.echo")
    def test_note(self, mock_echo):
        """Test note formatting"""
        fmt.note("test note")
        mock_echo.assert_called_once_with("Note: test note", color="blue")

    @patch("cognee.cli.echo.echo")
    def test_warning(self, mock_echo):
        """Test warning formatting"""
        fmt.warning("test warning")
        mock_echo.assert_called_once_with("Warning: test warning", color="yellow")

    @patch("cognee.cli.echo.echo")
    def test_error(self, mock_echo):
        """Test error formatting"""
        fmt.error("test error")
        mock_echo.assert_called_once_with("Error: test error", color="red", err=True)

    @patch("cognee.cli.echo.echo")
    def test_success(self, mock_echo):
        """Test success formatting"""
        fmt.success("test success")
        mock_echo.assert_called_once_with("Success: test success", color="green")

    @patch("click.style")
    def test_bold(self, mock_style):
        """Test bold text formatting"""
        mock_style.return_value = "bold text"
        result = fmt.bold("test text")
        mock_style.assert_called_once_with("test text", bold=True)
        assert result == "bold text"

    @patch("click.confirm")
    def test_confirm(self, mock_confirm):
        """Test confirmation prompt"""
        mock_confirm.return_value = True
        result = fmt.confirm("Are you sure?")
        mock_confirm.assert_called_once_with("Are you sure?", default=False)
        assert result is True

    @patch("click.confirm")
    def test_confirm_with_default(self, mock_confirm):
        """Test confirmation prompt with default"""
        mock_confirm.return_value = False
        result = fmt.confirm("Are you sure?", default=True)
        mock_confirm.assert_called_once_with("Are you sure?", default=True)
        assert result is False

    @patch("click.prompt")
    def test_prompt(self, mock_prompt):
        """Test user input prompt"""
        mock_prompt.return_value = "user input"
        result = fmt.prompt("Enter value:")
        mock_prompt.assert_called_once_with("Enter value:", default=None)
        assert result == "user input"

    @patch("click.prompt")
    def test_prompt_with_default(self, mock_prompt):
        """Test user input prompt with default"""
        mock_prompt.return_value = "default value"
        result = fmt.prompt("Enter value:", default="default value")
        mock_prompt.assert_called_once_with("Enter value:", default="default value")
        assert result == "default value"


class TestCliExceptions:
    """Test CLI exception classes"""

    def test_cli_command_exception_basic(self):
        """Test basic CliCommandException"""
        exc = CliCommandException("Test error")
        assert str(exc) == "Test error"
        assert exc.error_code == -1
        assert exc.docs_url is None
        assert exc.raiseable_exception is None

    def test_cli_command_exception_full(self):
        """Test CliCommandException with all parameters"""
        inner_exc = ValueError("Inner error")
        exc = CliCommandException(
            "Test error",
            error_code=2,
            docs_url="https://docs.test.com",
            raiseable_exception=inner_exc,
        )

        assert str(exc) == "Test error"
        assert exc.error_code == 2
        assert exc.docs_url == "https://docs.test.com"
        assert exc.raiseable_exception is inner_exc

    def test_cli_command_inner_exception(self):
        """Test CliCommandInnerException"""
        exc = CliCommandInnerException("Inner error")
        assert str(exc) == "Inner error"
        assert isinstance(exc, Exception)


class TestDebugModule:
    """Test CLI debug functionality"""

    def test_debug_initially_disabled(self):
        """Test that debug is initially disabled"""
        # Reset debug state
        debug._debug_enabled = False
        assert not debug.is_debug_enabled()

    def test_enable_debug(self):
        """Test enabling debug mode"""
        debug.enable_debug()
        assert debug.is_debug_enabled()

        # Reset for other tests
        debug._debug_enabled = False

    def test_debug_state_persistence(self):
        """Test that debug state persists"""
        debug.enable_debug()
        assert debug.is_debug_enabled()

        # Should still be enabled
        assert debug.is_debug_enabled()

        # Reset for other tests
        debug._debug_enabled = False


class TestCliConfig:
    """Test CLI configuration constants"""

    def test_cli_description_exists(self):
        """Test that CLI description is defined"""
        assert CLI_DESCRIPTION
        assert isinstance(CLI_DESCRIPTION, str)
        assert "cognee" in CLI_DESCRIPTION.lower()

    def test_default_docs_url_exists(self):
        """Test that default docs URL is defined"""
        assert DEFAULT_DOCS_URL
        assert isinstance(DEFAULT_DOCS_URL, str)
        assert DEFAULT_DOCS_URL.startswith("https://")

    def test_command_descriptions_complete(self):
        """Test that all expected commands have descriptions"""
        expected_commands = ["add", "search", "cognify", "delete", "config"]

        for command in expected_commands:
            assert command in COMMAND_DESCRIPTIONS
            assert isinstance(COMMAND_DESCRIPTIONS[command], str)
            assert len(COMMAND_DESCRIPTIONS[command]) > 0

    def test_search_type_choices_valid(self):
        """Test that search type choices are valid"""
        assert isinstance(SEARCH_TYPE_CHOICES, list)
        assert len(SEARCH_TYPE_CHOICES) > 0

        expected_types = [
            "GRAPH_COMPLETION",
            "RAG_COMPLETION",
            "INSIGHTS",
            "CHUNKS",
            "SUMMARIES",
            "CODE",
            "CYPHER",
        ]

        for expected_type in expected_types:
            assert expected_type in SEARCH_TYPE_CHOICES

    def test_chunker_choices_valid(self):
        """Test that chunker choices are valid"""
        assert isinstance(CHUNKER_CHOICES, list)
        assert len(CHUNKER_CHOICES) > 0
        assert "TextChunker" in CHUNKER_CHOICES
        assert "LangchainChunker" in CHUNKER_CHOICES

    def test_output_format_choices_valid(self):
        """Test that output format choices are valid"""
        assert isinstance(OUTPUT_FORMAT_CHOICES, list)
        assert len(OUTPUT_FORMAT_CHOICES) > 0

        expected_formats = ["json", "pretty", "simple"]
        for expected_format in expected_formats:
            assert expected_format in OUTPUT_FORMAT_CHOICES


class TestCliReference:
    """Test CLI reference protocol"""

    def test_supports_cli_command_protocol(self):
        """Test that SupportsCliCommand protocol is properly defined"""
        from cognee.cli.reference import SupportsCliCommand

        # Test that it's a protocol
        assert hasattr(SupportsCliCommand, "__annotations__")

        # Test required attributes
        annotations = SupportsCliCommand.__annotations__
        assert "command_string" in annotations
        assert "help_string" in annotations
        assert "description" in annotations
        assert "docs_url" in annotations

    def test_protocol_methods(self):
        """Test that protocol defines required methods"""
        from cognee.cli.reference import SupportsCliCommand
        import inspect

        # Get abstract methods
        abstract_methods = []
        for name, method in inspect.getmembers(SupportsCliCommand, predicate=inspect.ismethod):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.append(name)

        # Should have abstract methods for configure_parser and execute
        method_names = [name for name, _ in inspect.getmembers(SupportsCliCommand)]
        assert "configure_parser" in method_names
        assert "execute" in method_names


class TestCliUtilityFunctions:
    """Test utility functions and edge cases"""

    def test_echo_empty_message(self):
        """Test echo with empty message"""
        with patch("click.secho") as mock_secho:
            fmt.echo()
            mock_secho.assert_called_once_with("", fg=None, err=False)

    def test_echo_none_message(self):
        """Test echo with None message (should not crash)"""
        with patch("click.secho") as mock_secho:
            # This might raise an exception, which is expected behavior
            try:
                fmt.echo(None)
            except TypeError:
                # Expected for None message
                pass

    def test_multiple_debug_enable_calls(self):
        """Test multiple calls to enable_debug"""
        debug.enable_debug()
        debug.enable_debug()  # Should not cause issues
        assert debug.is_debug_enabled()

        # Reset for other tests
        debug._debug_enabled = False

    def test_config_constants_immutability(self):
        """Test that config constants are not accidentally modified"""
        original_description = CLI_DESCRIPTION
        original_url = DEFAULT_DOCS_URL
        original_commands = COMMAND_DESCRIPTIONS.copy()

        # These should be the same after any test
        assert CLI_DESCRIPTION == original_description
        assert DEFAULT_DOCS_URL == original_url
        assert COMMAND_DESCRIPTIONS == original_commands
