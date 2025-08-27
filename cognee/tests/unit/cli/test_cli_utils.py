"""
Tests for CLI utility functions and helper modules.
"""

from cognee.cli import debug
from cognee.cli.config import (
    CLI_DESCRIPTION,
    DEFAULT_DOCS_URL,
    COMMAND_DESCRIPTIONS,
    SEARCH_TYPE_CHOICES,
    CHUNKER_CHOICES,
    OUTPUT_FORMAT_CHOICES,
)
from cognee.cli._cognee import _discover_commands


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
        assert "cognee.ai" in DEFAULT_DOCS_URL

    def test_command_descriptions_complete(self):
        """Test that all expected commands have descriptions"""
        commands = _discover_commands()
        assert len(commands) > 0

        expected_commands = []
        for command_class in commands:
            command = command_class()
            expected_commands.append(command.command_string)

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

    def test_multiple_debug_enable_calls(self):
        """Test multiple calls to enable_debug"""
        debug.enable_debug()
        debug.enable_debug()  # Should not cause issues
        assert debug.is_debug_enabled() is True

        # Reset for other tests
        debug._debug_enabled = False
