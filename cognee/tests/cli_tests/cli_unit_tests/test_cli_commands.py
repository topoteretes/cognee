"""
Tests for individual CLI commands with proper mocking and coroutine handling.
"""

import pytest
import sys
import argparse
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, ANY
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

    @patch("cognee.cli.commands.add_command.asyncio.run", side_effect=_mock_run)
    def test_execute_single_item(self, mock_asyncio_run):
        """Test execute with single data item"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = AddCommand()
            args = argparse.Namespace(data=["test.txt"], dataset_name="test_dataset")
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.add.assert_awaited_once_with(data="test.txt", dataset_name="test_dataset")

    @patch("cognee.cli.commands.add_command.asyncio.run", side_effect=_mock_run)
    def test_execute_multiple_items(self, mock_asyncio_run):
        """Test execute with multiple data items"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = AddCommand()
            args = argparse.Namespace(data=["test1.txt", "test2.txt"], dataset_name="test_dataset")
            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.add.assert_awaited_once_with(
            data=["test1.txt", "test2.txt"], dataset_name="test_dataset"
        )

    @patch("cognee.cli.commands.add_command.asyncio.run")
    def test_execute_with_exception(self, mock_asyncio_run):
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

    @patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
    def test_execute_basic_search(self, mock_asyncio_run):
        """Test execute with basic search"""
        # Mock the cognee module and SearchType
        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(return_value=["result1", "result2"])
        mock_search_type = MagicMock()
        mock_search_type.__getitem__.return_value = "GRAPH_COMPLETION"

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = SearchCommand()
            args = argparse.Namespace(
                query_text="test query",
                query_type="GRAPH_COMPLETION",
                datasets=None,
                top_k=10,
                system_prompt=None,
                output_format="pretty",
            )
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
        assert called_enum.name == "GRAPH_COMPLETION"

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

    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_execute_basic_cognify(self, mock_asyncio_run):
        """Test execute with basic cognify"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock(return_value="success")

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=None,
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
    @patch("cognee.cli.commands.delete_command.asyncio.run", side_effect=_mock_run)
    def test_execute_delete_dataset_with_confirmation(self, mock_asyncio_run, mock_confirm):
        """Test execute delete dataset with user confirmation"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.delete = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = DeleteCommand()
            args = argparse.Namespace(
                dataset_name="test_dataset", user_id=None, all=False, force=False
            )

            mock_confirm.return_value = True

            command.execute(args)

        mock_confirm.assert_called_once_with(f"Delete dataset '{args.dataset_name}'?")
        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.delete.assert_awaited_once_with(dataset_name="test_dataset", user_id=None)

    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    def test_execute_delete_cancelled(self, mock_confirm):
        """Test execute when user cancels deletion"""
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=False)

        mock_confirm.return_value = False

        # Should not raise exception, just return
        command.execute(args)

        mock_confirm.assert_called_once_with(f"Delete dataset '{args.dataset_name}'?")

    @patch("cognee.cli.commands.delete_command.asyncio.run", side_effect=_mock_run)
    def test_execute_delete_forced(self, mock_asyncio_run):
        """Test execute delete with force flag"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.delete = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = DeleteCommand()
            args = argparse.Namespace(
                dataset_name="test_dataset", user_id=None, all=False, force=True
            )

            command.execute(args)

        mock_asyncio_run.assert_called_once()
        assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])
        mock_cognee.delete.assert_awaited_once_with(dataset_name="test_dataset", user_id=None)

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

    @patch("builtins.__import__")
    def test_execute_get_action(self, mock_import):
        """Test execute get action"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.config.get = MagicMock(return_value="openai")
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        args = argparse.Namespace(config_action="get", key="llm_provider")

        command.execute(args)

    @patch("builtins.__import__")
    def test_execute_set_action(self, mock_import):
        """Test execute set action"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.config.set = MagicMock()
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        args = argparse.Namespace(config_action="set", key="llm_provider", value="anthropic")

        command.execute(args)

    @patch("builtins.__import__")
    def test_execute_set_action_json_value(self, mock_import):
        """Test execute set action with JSON value"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.config.set = MagicMock()
        mock_import.return_value = mock_cognee

        command = ConfigCommand()
        args = argparse.Namespace(config_action="set", key="chunk_size", value="1024")

        command.execute(args)

    def test_execute_list_action(self):
        """Test execute list action"""
        command = ConfigCommand()
        args = argparse.Namespace(config_action="list")

        # Should not raise exception
        command.execute(args)

    @patch("cognee.cli.commands.config_command.fmt.confirm")
    def test_execute_unset_action(self, mock_confirm):
        """Test execute unset action"""
        # Mock the cognee module
        mock_cognee = MagicMock()
        mock_cognee.config.set_llm_provider = MagicMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = ConfigCommand()
            args = argparse.Namespace(config_action="unset", key="llm_provider", force=False)

            mock_confirm.return_value = True

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
        # Test with an invalid action that will cause an exception in the main execute method
        command = ConfigCommand()
        args = argparse.Namespace(config_action="invalid_action")

        # This should not raise CliCommandException, just handle it gracefully
        # The config command handles unknown actions by showing an error message
        command.execute(args)
