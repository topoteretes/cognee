"""
Tests for individual CLI commands.
"""

import pytest
import argparse
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from cognee.cli.commands.add_command import AddCommand
from cognee.cli.commands.search_command import SearchCommand
from cognee.cli.commands.cognify_command import CognifyCommand
from cognee.cli.commands.delete_command import DeleteCommand
from cognee.cli.commands.config_command import ConfigCommand
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class TestAddCommand:
    """Test the AddCommand class"""

    def test_command_properties(self):
        """Test basic command properties"""
        command = AddCommand()
        assert command.command_string == "add"
        assert "Add data" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        """Test parser configuration"""
        command = AddCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        # Check that required arguments are added
        actions = {action.dest: action for action in parser._actions}
        assert "data" in actions
        assert "dataset_name" in actions

        # Check data argument accepts multiple values
        assert actions["data"].nargs == "+"

    @patch("cognee.cli.commands.add_command.asyncio.run")
    @patch("cognee.cli.commands.add_command.cognee")
    def test_execute_single_item(self, mock_cognee, mock_asyncio_run):
        """Test execute with single data item"""
        command = AddCommand()
        args = argparse.Namespace(data=["test.txt"], dataset_name="test_dataset")

        mock_cognee.add = AsyncMock()

        command.execute(args)

        mock_asyncio_run.assert_called_once()
        # Check that the async function would be called correctly
        assert mock_asyncio_run.call_args[0][0]  # async function was passed

    @patch("cognee.cli.commands.add_command.asyncio.run")
    @patch("cognee.cli.commands.add_command.cognee")
    def test_execute_multiple_items(self, mock_cognee, mock_asyncio_run):
        """Test execute with multiple data items"""
        command = AddCommand()
        args = argparse.Namespace(data=["test1.txt", "test2.txt"], dataset_name="test_dataset")

        mock_cognee.add = AsyncMock()

        command.execute(args)

        mock_asyncio_run.assert_called_once()

    @patch("cognee.cli.commands.add_command.asyncio.run")
    @patch("cognee.cli.commands.add_command.cognee")
    def test_execute_with_exception(self, mock_cognee, mock_asyncio_run):
        """Test execute handles exceptions properly"""
        command = AddCommand()
        args = argparse.Namespace(data=["test.txt"], dataset_name="test_dataset")

        mock_asyncio_run.side_effect = Exception("Test error")

        with pytest.raises(CliCommandException):
            command.execute(args)


class TestSearchCommand:
    """Test the SearchCommand class"""

    def test_command_properties(self):
        """Test basic command properties"""
        command = SearchCommand()
        assert command.command_string == "search"
        assert "Search and query" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        """Test parser configuration"""
        command = SearchCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        # Check that required arguments are added
        actions = {action.dest: action for action in parser._actions}
        assert "query_text" in actions
        assert "query_type" in actions
        assert "datasets" in actions
        assert "top_k" in actions
        assert "output_format" in actions

        # Check default values
        assert actions["query_type"].default == "GRAPH_COMPLETION"
        assert actions["top_k"].default == 10
        assert actions["output_format"].default == "pretty"

    @patch("cognee.cli.commands.search_command.asyncio.run")
    @patch("cognee.cli.commands.search_command.cognee")
    def test_execute_basic_search(self, mock_cognee, mock_asyncio_run):
        """Test execute with basic search"""
        command = SearchCommand()
        args = argparse.Namespace(
            query_text="test query",
            query_type="GRAPH_COMPLETION",
            datasets=None,
            top_k=10,
            system_prompt=None,
            output_format="pretty",
        )

        mock_cognee.search = AsyncMock(return_value=["result1", "result2"])
        mock_asyncio_run.return_value = ["result1", "result2"]

        command.execute(args)

        mock_asyncio_run.assert_called_once()

    @patch("cognee.cli.commands.search_command.asyncio.run")
    @patch("cognee.cli.commands.search_command.cognee")
    def test_execute_json_output(self, mock_cognee, mock_asyncio_run):
        """Test execute with JSON output format"""
        command = SearchCommand()
        args = argparse.Namespace(
            query_text="test query",
            query_type="CHUNKS",
            datasets=["dataset1"],
            top_k=5,
            system_prompt=None,
            output_format="json",
        )

        mock_cognee.search = AsyncMock(return_value=[{"chunk": "test"}])
        mock_asyncio_run.return_value = [{"chunk": "test"}]

        command.execute(args)

        mock_asyncio_run.assert_called_once()

    @patch("cognee.cli.commands.search_command.asyncio.run")
    def test_execute_with_exception(self, mock_asyncio_run):
        """Test execute handles exceptions properly"""
        command = SearchCommand()
        args = argparse.Namespace(
            query_text="test query",
            query_type="GRAPH_COMPLETION",
            datasets=None,
            top_k=10,
            system_prompt=None,
            output_format="pretty",
        )

        mock_asyncio_run.side_effect = Exception("Search error")

        with pytest.raises(CliCommandException):
            command.execute(args)


class TestCognifyCommand:
    """Test the CognifyCommand class"""

    def test_command_properties(self):
        """Test basic command properties"""
        command = CognifyCommand()
        assert command.command_string == "cognify"
        assert "Transform ingested data" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        """Test parser configuration"""
        command = CognifyCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        # Check that arguments are added
        actions = {action.dest: action for action in parser._actions}
        assert "datasets" in actions
        assert "chunk_size" in actions
        assert "ontology_file" in actions
        assert "chunker" in actions
        assert "background" in actions
        assert "verbose" in actions

        # Check default values
        assert actions["chunker"].default == "TextChunker"

    @patch("cognee.cli.commands.cognify_command.asyncio.run")
    @patch("cognee.cli.commands.cognify_command.cognee")
    def test_execute_basic_cognify(self, mock_cognee, mock_asyncio_run):
        """Test execute with basic cognify"""
        command = CognifyCommand()
        args = argparse.Namespace(
            datasets=None,
            chunk_size=None,
            ontology_file=None,
            chunker="TextChunker",
            background=False,
            verbose=False,
        )

        mock_cognee.cognify = AsyncMock(return_value="success")
        mock_asyncio_run.return_value = "success"

        command.execute(args)

        mock_asyncio_run.assert_called_once()

    @patch("cognee.cli.commands.cognify_command.asyncio.run")
    @patch("cognee.cli.commands.cognify_command.cognee")
    def test_execute_background_mode(self, mock_cognee, mock_asyncio_run):
        """Test execute with background mode"""
        command = CognifyCommand()
        args = argparse.Namespace(
            datasets=["dataset1"],
            chunk_size=1024,
            ontology_file="/path/to/ontology.owl",
            chunker="LangchainChunker",
            background=True,
            verbose=True,
        )

        mock_cognee.cognify = AsyncMock(return_value="background_started")
        mock_asyncio_run.return_value = "background_started"

        command.execute(args)

        mock_asyncio_run.assert_called_once()

    @patch("cognee.cli.commands.cognify_command.asyncio.run")
    def test_execute_with_exception(self, mock_asyncio_run):
        """Test execute handles exceptions properly"""
        command = CognifyCommand()
        args = argparse.Namespace(
            datasets=None,
            chunk_size=None,
            ontology_file=None,
            chunker="TextChunker",
            background=False,
            verbose=False,
        )

        mock_asyncio_run.side_effect = Exception("Cognify error")

        with pytest.raises(CliCommandException):
            command.execute(args)


class TestDeleteCommand:
    """Test the DeleteCommand class"""

    def test_command_properties(self):
        """Test basic command properties"""
        command = DeleteCommand()
        assert command.command_string == "delete"
        assert "Delete data" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        """Test parser configuration"""
        command = DeleteCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        # Check that arguments are added
        actions = {action.dest: action for action in parser._actions}
        assert "dataset_name" in actions
        assert "user_id" in actions
        assert "all" in actions
        assert "force" in actions

    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    @patch("cognee.cli.commands.delete_command.asyncio.run")
    @patch("cognee.cli.commands.delete_command.cognee")
    def test_execute_delete_dataset_with_confirmation(
        self, mock_cognee, mock_asyncio_run, mock_confirm
    ):
        """Test execute delete dataset with user confirmation"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=False)

        mock_confirm.return_value = True
        mock_cognee.delete = AsyncMock()

        command.execute(args)

        mock_confirm.assert_called_once()
        mock_asyncio_run.assert_called_once()

    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    def test_execute_delete_cancelled(self, mock_confirm):
        """Test execute when user cancels deletion"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=False)

        mock_confirm.return_value = False

        # Should not raise exception, just return
        command.execute(args)

        mock_confirm.assert_called_once()

    @patch("cognee.cli.commands.delete_command.asyncio.run")
    @patch("cognee.cli.commands.delete_command.cognee")
    def test_execute_delete_forced(self, mock_cognee, mock_asyncio_run):
        """Test execute delete with force flag"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=True)

        mock_cognee.delete = AsyncMock()

        command.execute(args)

        mock_asyncio_run.assert_called_once()

    def test_execute_no_delete_target(self):
        """Test execute when no delete target is specified"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name=None, user_id=None, all=False, force=False)

        # Should not raise exception, just return with error message
        command.execute(args)

    @patch("cognee.cli.commands.delete_command.asyncio.run")
    def test_execute_with_exception(self, mock_asyncio_run):
        """Test execute handles exceptions properly"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=True)

        mock_asyncio_run.side_effect = Exception("Delete error")

        with pytest.raises(CliCommandException):
            command.execute(args)


class TestConfigCommand:
    """Test the ConfigCommand class"""

    def test_command_properties(self):
        """Test basic command properties"""
        command = ConfigCommand()
        assert command.command_string == "config"
        assert "Manage cognee configuration" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        """Test parser configuration"""
        command = ConfigCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        # Check that subparsers are created
        subparsers_actions = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ]
        assert len(subparsers_actions) == 1

        subparsers = subparsers_actions[0]
        assert "get" in subparsers.choices
        assert "set" in subparsers.choices
        assert "list" in subparsers.choices
        assert "unset" in subparsers.choices
        assert "reset" in subparsers.choices

    def test_execute_no_action(self):
        """Test execute when no config action is provided"""
        command = ConfigCommand()
        args = argparse.Namespace()

        # Should not raise exception, just return with error message
        command.execute(args)

    @patch("cognee.cli.commands.config_command.cognee")
    def test_execute_get_action(self, mock_cognee):
        """Test execute get action"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="get", key="llm_provider")

        mock_cognee.config.get = MagicMock(return_value="openai")

        command.execute(args)

        # Should call get method if available
        if hasattr(mock_cognee.config, "get"):
            mock_cognee.config.get.assert_called_with("llm_provider")

    @patch("cognee.cli.commands.config_command.cognee")
    def test_execute_set_action(self, mock_cognee):
        """Test execute set action"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="set", key="llm_provider", value="anthropic")

        mock_cognee.config.set = MagicMock()

        command.execute(args)

        mock_cognee.config.set.assert_called_with("llm_provider", "anthropic")

    @patch("cognee.cli.commands.config_command.cognee")
    def test_execute_set_action_json_value(self, mock_cognee):
        """Test execute set action with JSON value"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="set", key="chunk_size", value="1024")

        mock_cognee.config.set = MagicMock()

        command.execute(args)

        # Value should be parsed as string since it's not valid JSON
        mock_cognee.config.set.assert_called_with("chunk_size", "1024")

    def test_execute_list_action(self):
        """Test execute list action"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="list")

        # Should not raise exception
        command.execute(args)

    @patch("cognee.cli.commands.config_command.fmt.confirm")
    @patch("cognee.cli.commands.config_command.cognee")
    def test_execute_unset_action(self, mock_cognee, mock_confirm):
        """Test execute unset action"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="unset", key="llm_provider", force=False)

        mock_confirm.return_value = True
        mock_cognee.config.set_llm_provider = MagicMock()

        command.execute(args)

        mock_confirm.assert_called_once()

    @patch("cognee.cli.commands.config_command.fmt.confirm")
    def test_execute_reset_action(self, mock_confirm):
        """Test execute reset action"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="reset", force=False)

        mock_confirm.return_value = True

        # Should not raise exception
        command.execute(args)

        mock_confirm.assert_called_once()

    def test_execute_with_exception(self):
        """Test execute handles exceptions properly"""
        command = ConfigCommand()
        # Create args that will cause an exception in _handle_set
        args = argparse.Namespace(config_action="set", key="invalid_key", value="value")

        with patch("cognee.cli.commands.config_command.cognee") as mock_cognee:
            mock_cognee.config.set.side_effect = Exception("Config error")

            with pytest.raises(CliCommandException):
                command.execute(args)
