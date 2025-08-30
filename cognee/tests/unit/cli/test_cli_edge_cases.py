"""
Tests for CLI edge cases and error scenarios with proper mocking.
"""

import pytest
import sys
import asyncio
import argparse
from unittest.mock import patch, MagicMock, AsyncMock, ANY, call
from cognee.cli.commands.add_command import AddCommand
from cognee.cli.commands.search_command import SearchCommand
from cognee.cli.commands.cognify_command import CognifyCommand
from cognee.cli.commands.delete_command import DeleteCommand
from cognee.cli.commands.config_command import ConfigCommand
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


# Mock asyncio.run to properly handle coroutines
def _mock_run(coro):
    # Create an event loop and run the coroutine
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestAddCommandEdgeCases:
    """Test edge cases for AddCommand"""

    @patch("cognee.cli.commands.add_command.asyncio.run", side_effect=_mock_run)
    def test_add_empty_data_list(self, mock_asyncio_run):
        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = AddCommand()
            args = argparse.Namespace(data=[], dataset_name="test_dataset")
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.add.assert_awaited_once_with(data=[], dataset_name="test_dataset")

    @patch("cognee.cli.commands.add_command.asyncio.run")
    def test_add_asyncio_run_exception(self, mock_asyncio_run):
        """Test add command when asyncio.run itself fails"""
        command = AddCommand()
        args = argparse.Namespace(data=["test.txt"], dataset_name="test_dataset")

        mock_asyncio_run.side_effect = RuntimeError("Event loop error")

        with pytest.raises(CliCommandException):
            command.execute(args)

    def test_add_special_characters_in_data(self):
        """Test add command with special characters in file paths"""
        command = AddCommand()

        # Create parser to test argument parsing with special characters
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Test parsing with special characters
        special_paths = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.dots.txt",
        ]

        args = parser.parse_args(special_paths + ["--dataset-name", "test"])
        assert args.data == special_paths
        assert args.dataset_name == "test"


class TestSearchCommandEdgeCases:
    """Test edge cases for SearchCommand"""

    @patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
    def test_search_empty_results(self, mock_asyncio_run):
        """Test search command with empty results"""
        # Mock the cognee module and SearchType
        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(return_value=[])
        mock_search_type = MagicMock()
        mock_search_type.__getitem__.return_value = "GRAPH_COMPLETION"

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = SearchCommand()
            args = argparse.Namespace(
                query_text="nonexistent query",
                query_type="GRAPH_COMPLETION",
                datasets=None,
                top_k=10,
                system_prompt=None,
                output_format="pretty",
            )

            # Should handle empty results gracefully
            command.execute(args)

        mock_asyncio_run.return_value = []
        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.search.assert_awaited_once_with(
            query_text="nonexistent query",
            query_type=ANY,
            datasets=None,
            top_k=10,
            system_prompt_path="answer_simple_question.txt",
        )
        # verify the enum’s name separately
        called_enum = mock_cognee.search.await_args.kwargs["query_type"]
        assert called_enum.name == "GRAPH_COMPLETION"

    @patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
    def test_search_very_large_top_k(self, mock_asyncio_run):
        """Test search command with very large top-k value"""
        # Mock the cognee module and SearchType
        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(return_value=["result1"])
        mock_search_type = MagicMock()
        mock_search_type.__getitem__.return_value = "CHUNKS"

        mock_asyncio_run.return_value = ["result1"]

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = SearchCommand()
            args = argparse.Namespace(
                query_text="test query",
                query_type="CHUNKS",
                datasets=None,
                top_k=999999,  # Very large value
                system_prompt=None,
                output_format="json",
            )

            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.search.assert_awaited_once_with(
            query_text="test query",
            query_type=ANY,
            datasets=None,
            top_k=999999,
            system_prompt_path="answer_simple_question.txt",
        )
        # verify the enum’s name separately
        called_enum = mock_cognee.search.await_args.kwargs["query_type"]
        assert called_enum.name == "CHUNKS"

    @patch("builtins.__import__")
    def test_search_invalid_search_type_enum(self, mock_import):
        """Test search command with invalid SearchType enum conversion"""
        # Mock SearchType to raise KeyError
        mock_search_type = MagicMock()
        mock_search_type.__getitem__.side_effect = KeyError("INVALID_TYPE")

        def mock_import_func(name, fromlist=None, *args, **kwargs):
            if name == "cognee.modules.search.types":
                module = MagicMock()
                module.SearchType = mock_search_type
                return module
            return MagicMock()

        mock_import.side_effect = mock_import_func

        command = SearchCommand()
        args = argparse.Namespace(
            query_text="test query",
            query_type="INVALID_TYPE",  # This would fail enum conversion
            datasets=None,
            top_k=10,
            system_prompt=None,
            output_format="pretty",
        )

        with pytest.raises(CliCommandException):
            command.execute(args)

    def test_search_unicode_query(self):
        """Test search command with unicode characters in query"""
        command = SearchCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        unicode_query = "测试查询 🔍 émojis and spéciál chars"
        args = parser.parse_args([unicode_query])
        assert args.query_text == unicode_query

    @patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
    def test_search_results_with_none_values(self, mock_asyncio_run):
        """Test search command when results contain None values"""
        # Mock the cognee module and SearchType
        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(return_value=[None, "valid result", None])
        mock_search_type = MagicMock()
        mock_search_type.__getitem__.return_value = "CHUNKS"

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = SearchCommand()
            args = argparse.Namespace(
                query_text="test query",
                query_type="CHUNKS",
                datasets=None,
                top_k=10,
                system_prompt=None,
                output_format="pretty",
            )

            # Should handle None values gracefully
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.search.assert_awaited_once_with(
            query_text="test query",
            query_type=ANY,
            datasets=None,
            top_k=10,
            system_prompt_path="answer_simple_question.txt",
        )
        # verify the enum’s name separately
        called_enum = mock_cognee.search.await_args.kwargs["query_type"]
        assert called_enum.name == "CHUNKS"


class TestCognifyCommandEdgeCases:
    """Test edge cases for CognifyCommand"""

    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_cognify_invalid_chunk_size(self, mock_asyncio_run):
        """Test cognify command with invalid chunk size"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=None,
                chunk_size=-100,  # Invalid negative chunk size
                ontology_file=None,
                chunker="TextChunker",
                background=False,
                verbose=False,
            )

            # Should pass the invalid value to cognify and let it handle the validation
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        from cognee.modules.chunking.TextChunker import TextChunker

        mock_cognee.cognify.assert_awaited_once_with(
            datasets=None,
            chunk_size=-100,
            ontology_file_path=None,
            chunker=TextChunker,
            run_in_background=False,
        )

    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_cognify_nonexistent_ontology_file(self, mock_asyncio_run):
        """Test cognify command with nonexistent ontology file"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=None,
                chunk_size=None,
                ontology_file="/nonexistent/path/ontology.owl",
                chunker="TextChunker",
                background=False,
                verbose=False,
            )

            # Should pass the path to cognify and let it handle file validation
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        from cognee.modules.chunking.TextChunker import TextChunker

        mock_cognee.cognify.assert_awaited_once_with(
            datasets=None,
            chunk_size=None,
            ontology_file_path="/nonexistent/path/ontology.owl",
            chunker=TextChunker,
            run_in_background=False,
        )

    @patch("cognee.cli.commands.cognify_command.asyncio.run")
    def test_cognify_langchain_chunker_import_error(self, mock_asyncio_run):
        """Test cognify command when LangchainChunker import fails"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock()

        def mock_import_func(name, fromlist=None, *args, **kwargs):
            if name == "cognee":
                return mock_cognee
            elif (
                name == "cognee.modules.chunking.LangchainChunker"
                and fromlist
                and "LangchainChunker" in fromlist
            ):
                raise ImportError("LangchainChunker not available")
            elif (
                name == "cognee.modules.chunking.TextChunker"
                and fromlist
                and "TextChunker" in fromlist
            ):
                module = MagicMock()
                module.TextChunker = MagicMock()
                return module
            return MagicMock()

        with (
            patch("builtins.__import__", side_effect=mock_import_func),
            patch.dict(sys.modules, {"cognee": mock_cognee}),
        ):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=None,
                chunk_size=None,
                ontology_file=None,
                chunker="LangchainChunker",
                background=False,
                verbose=True,
            )

            # Should fall back to TextChunker and show warning
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])

    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_cognify_empty_datasets_list(self, mock_asyncio_run):
        """Test cognify command with nonexistent ontology file"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=[],
                chunk_size=None,
                ontology_file=None,
                chunker="TextChunker",
                background=False,
                verbose=False,
            )

            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        from cognee.modules.chunking.TextChunker import TextChunker

        mock_cognee.cognify.assert_awaited_once_with(
            datasets=None,
            chunk_size=None,
            ontology_file_path=None,
            chunker=TextChunker,
            run_in_background=False,
        )


class TestDeleteCommandEdgeCases:
    """Test edge cases for DeleteCommand"""

    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    @patch("cognee.cli.commands.delete_command.asyncio.run", side_effect=_mock_run)
    def test_delete_all_with_user_id(self, mock_asyncio_run, mock_confirm):
        """Test delete command with both --all and --user-id"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.delete = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = DeleteCommand()
            args = argparse.Namespace(dataset_name=None, user_id="test_user", all=True, force=False)

            mock_confirm.return_value = True

            # Should handle both flags being set
            command.execute(args)

        mock_confirm.assert_called_once_with("Delete ALL data from cognee?")
        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.delete.assert_awaited_once_with(dataset_name=None, user_id="test_user")

    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    def test_delete_confirmation_keyboard_interrupt(self, mock_confirm):
        """Test delete command when user interrupts confirmation"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=False)

        mock_confirm.side_effect = KeyboardInterrupt()

        # Should handle KeyboardInterrupt gracefully
        with pytest.raises(KeyboardInterrupt):
            command.execute(args)

    @patch("cognee.cli.commands.delete_command.asyncio.run")
    def test_delete_async_exception_handling(self, mock_asyncio_run):
        """Test delete command async exception handling"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=True)

        # Mock asyncio.run to raise exception directly
        mock_asyncio_run.side_effect = ValueError("Database connection failed")

        with pytest.raises(CliCommandException):
            command.execute(args)

    def test_delete_special_characters_in_dataset_name(self):
        """Test delete command with special characters in dataset name"""
        command = DeleteCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        special_names = [
            "dataset with spaces",
            "dataset-with-dashes",
            "dataset_with_underscores",
            "dataset.with.dots",
            "dataset/with/slashes",
        ]

        for name in special_names:
            args = parser.parse_args(["--dataset-name", name])
            assert args.dataset_name == name


class TestConfigCommandEdgeCases:
    """Test edge cases for ConfigCommand"""

    def test_config_no_subcommand_specified(self):
        """Test config command when no subcommand is specified"""
        command = ConfigCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        # Parse with no subcommand - should set config_action to None
        args = parser.parse_args([])
        assert not hasattr(args, "config_action") or args.config_action is None

    @patch("builtins.__import__")
    def test_config_get_nonexistent_key(self, mock_import):
        """Test config get with nonexistent key"""
        # Mock config.get to raise exception for nonexistent key
        mock_cognee = MagicMock()
        mock_cognee.config.get = MagicMock(side_effect=KeyError("Key not found"))
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        args = argparse.Namespace(config_action="get", key="nonexistent_key")

        # Should handle the exception gracefully
        command.execute(args)
        mock_cognee.config.get.assert_called_once_with("nonexistent_key")

    @patch("builtins.__import__")
    def test_config_set_complex_json_value(self, mock_import):
        """Test config set with complex JSON value"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.config.set = MagicMock()
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        complex_json = '{"nested": {"key": "value"}, "array": [1, 2, 3]}'
        complex_json_expected_value = {"nested": {"key": "value"}, "array": [1, 2, 3]}
        args = argparse.Namespace(config_action="set", key="complex_config", value=complex_json)

        command.execute(args)
        mock_cognee.config.set.assert_called_once_with(
            "complex_config", complex_json_expected_value
        )

    @patch("builtins.__import__")
    def test_config_set_invalid_json_value(self, mock_import):
        """Test config set with invalid JSON value"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.config.set = MagicMock()
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        invalid_json = '{"invalid": json}'
        args = argparse.Namespace(config_action="set", key="test_key", value=invalid_json)

        command.execute(args)
        mock_cognee.config.set.assert_called_once_with("test_key", invalid_json)

    @patch("cognee.cli.commands.config_command.fmt.confirm")
    def test_config_unset_unknown_key(self, mock_confirm):
        """Test config unset with unknown key"""
        # Mock the cognee module
        mock_cognee = MagicMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = ConfigCommand()
            args = argparse.Namespace(config_action="unset", key="unknown_key", force=False)

            mock_confirm.return_value = True

            # Should show error for unknown key
            command.execute(args)

        mock_confirm.assert_called_once()

    @patch("builtins.__import__")
    def test_config_unset_method_not_found(self, mock_import):
        """Test config unset when method doesn't exist on config object"""
        # Mock config object without the expected method
        mock_cognee = MagicMock()
        mock_cognee.config = MagicMock()
        # Don't set the set_llm_provider method
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        args = argparse.Namespace(config_action="unset", key="llm_provider", force=True)

        # Should handle AttributeError gracefully
        command.execute(args)
        mock_cognee.config.unset.assert_not_called()

    def test_config_invalid_subcommand(self):
        """Test config command with invalid subcommand"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="invalid_action")

        # Should handle unknown subcommand gracefully
        command.execute(args)


class TestGeneralEdgeCases:
    """Test general edge cases that apply to multiple commands"""

    def test_command_with_none_args(self):
        """Test command execution with None args"""
        commands = [
            AddCommand(),
            SearchCommand(),
            CognifyCommand(),
            DeleteCommand(),
            ConfigCommand(),
        ]

        for command in commands:
            # Should not crash with None args, though it might raise exceptions
            try:
                command.execute(None)
            except (AttributeError, CliCommandException):
                # Expected behavior for None args
                pass

    def test_parser_configuration_with_none_parser(self):
        """Test parser configuration with None parser"""
        commands = [
            AddCommand(),
            SearchCommand(),
            CognifyCommand(),
            DeleteCommand(),
            ConfigCommand(),
        ]

        for command in commands:
            # Should not crash, though it might raise AttributeError
            try:
                command.configure_parser(None)
            except AttributeError:
                # Expected behavior for None parser
                pass

    def test_command_properties_are_strings(self):
        """Test that all command properties are proper strings"""
        commands = [
            AddCommand(),
            SearchCommand(),
            CognifyCommand(),
            DeleteCommand(),
            ConfigCommand(),
        ]

        for command in commands:
            assert isinstance(command.command_string, str)
            assert len(command.command_string) > 0

            assert isinstance(command.help_string, str)
            assert len(command.help_string) > 0

            if hasattr(command, "description") and command.description:
                assert isinstance(command.description, str)

            if hasattr(command, "docs_url") and command.docs_url:
                assert isinstance(command.docs_url, str)

    @patch("tempfile.NamedTemporaryFile")
    def test_commands_with_temp_files(self, mock_temp_file):
        """Test commands that might work with temporary files"""
        # Mock a temporary file
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_file.txt"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Test AddCommand with temp file
        command = AddCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        args = parser.parse_args([mock_file.name])
        assert args.data == [mock_file.name]
