"""
Tests for individual CLI commands with proper mocking and coroutine handling.
"""

import os
import pytest
import sys
import argparse
import asyncio
from uuid import uuid4
from unittest.mock import patch, MagicMock, AsyncMock, ANY
import cognee
from cognee.cli.commands.add_command import AddCommand
from cognee.cli.commands.search_command import SearchCommand
from cognee.cli.commands.recall_command import RecallCommand
from cognee.cli.commands.cognify_command import CognifyCommand
from cognee.cli.commands.delete_command import DeleteCommand
from cognee.cli.commands.forget_command import ForgetCommand
from cognee.cli.commands.config_command import ConfigCommand
from cognee.cli.exceptions import CliCommandException
from cognee.modules.data.methods.get_deletion_counts import DeletionCountsPreview
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.models import User


# Mock asyncio.run to properly handle coroutines
def _mock_run(coro):
    # Create an event loop and run the coroutine
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_user():
    """Return a fake user with an id for resolve_cli_user mocks."""
    u = MagicMock()
    u.id = uuid4()
    return u


# Patch resolve_cli_user globally for all tests that use _mock_run,
# so they don't hit the real database.
_RESOLVE_USER_PATCH = "cognee.cli.user_resolution.resolve_cli_user"


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

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.add_command.asyncio.run", side_effect=_mock_run)
    def test_execute_single_item(self, mock_asyncio_run, _mock_resolve):
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
        mock_cognee.add.assert_awaited_once_with(
            data="test.txt", dataset_name="test_dataset", user=ANY
        )

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.add_command.asyncio.run", side_effect=_mock_run)
    def test_execute_multiple_items(self, mock_asyncio_run, _mock_resolve):
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
            data=["test1.txt", "test2.txt"], dataset_name="test_dataset", user=ANY
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
        assert actions["query_type"].default == "HYBRID_COMPLETION"
        assert actions["top_k"].default == 10
        assert actions["output_format"].default == "pretty"

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
    def test_execute_basic_search(self, mock_asyncio_run, _mock_resolve):
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
            user=ANY,
            datasets=None,
            top_k=10,
            system_prompt_path="answer_simple_question.txt",
            session_id=ANY,
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


class TestRecallCommand:
    def test_configure_parser_accepts_hybrid_and_omits_query_type_by_default(self):
        command = RecallCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)
        actions = {action.dest: action for action in parser._actions}

        assert actions["query_type"].default is None
        assert "HYBRID_COMPLETION" in actions["query_type"].choices

    @patch("cognee.cli.commands.recall_command.asyncio.run", side_effect=_mock_run)
    def test_session_only_when_query_type_is_omitted(self, mock_asyncio_run):
        mock_cognee = MagicMock()
        mock_cognee.recall = AsyncMock(
            return_value=[{"_source": "session", "question": "q", "answer": "a"}]
        )

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = RecallCommand()
            args = argparse.Namespace(
                query_text="test query",
                query_type=None,
                datasets=None,
                top_k=10,
                system_prompt=None,
                session_id="sess",
                output_format="pretty",
            )
            command.execute(args)

        mock_cognee.recall.assert_awaited_once()
        kwargs = mock_cognee.recall.await_args.kwargs
        assert "query_type" not in kwargs
        assert kwargs["session_id"] == "sess"

    @patch("cognee.cli.commands.recall_command.asyncio.run", side_effect=_mock_run)
    def test_explicit_hybrid_with_session_is_not_session_only(self, mock_asyncio_run):
        mock_cognee = MagicMock()
        mock_cognee.recall = AsyncMock(return_value=["answer"])

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = RecallCommand()
            args = argparse.Namespace(
                query_text="test query",
                query_type="HYBRID_COMPLETION",
                datasets=None,
                top_k=10,
                system_prompt=None,
                session_id="sess",
                output_format="pretty",
            )
            command.execute(args)

        kwargs = mock_cognee.recall.await_args.kwargs
        assert kwargs["query_type"].name == "HYBRID_COMPLETION"
        assert kwargs["session_id"] == "sess"

    @patch("cognee.cli.commands.recall_command.asyncio.run", side_effect=_mock_run)
    def test_explicit_graph_completion_with_session_is_not_session_only(self, mock_asyncio_run):
        mock_cognee = MagicMock()
        mock_cognee.recall = AsyncMock(return_value=["answer"])

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = RecallCommand()
            args = argparse.Namespace(
                query_text="test query",
                query_type="GRAPH_COMPLETION",
                datasets=None,
                top_k=10,
                system_prompt=None,
                session_id="sess",
                output_format="pretty",
            )
            command.execute(args)

        kwargs = mock_cognee.recall.await_args.kwargs
        assert kwargs["query_type"].name == "GRAPH_COMPLETION"
        assert kwargs["session_id"] == "sess"


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

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_execute_basic_cognify(self, mock_asyncio_run, _mock_resolve):
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
            user=ANY,
            chunk_size=None,
            config=None,
            chunker=TextChunker,
            run_in_background=False,
            chunks_per_batch=None,
            dry_run=False,
        )

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_execute_with_ontology_file(self, mock_asyncio_run, _mock_resolve, tmp_path):
        """--ontology-file is translated into the canonical ontology Config structure"""
        ontology_path = tmp_path / "ontology.owl"
        ontology_path.write_text(
            '<?xml version="1.0"?>'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF>'
        )

        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock(return_value="success")

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=None,
                chunk_size=None,
                ontology_file=str(ontology_path),
                chunker="TextChunker",
                background=False,
                verbose=False,
            )
            command.execute(args)

        from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

        config = mock_cognee.cognify.await_args.kwargs["config"]
        resolver = config["ontology_config"]["ontology_resolver"]
        assert isinstance(resolver, RDFLibOntologyResolver)
        assert resolver.ontology_file == str(ontology_path)

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_execute_with_multiple_ontology_files(self, mock_asyncio_run, _mock_resolve, tmp_path):
        """A comma-separated --ontology-file loads all listed files into the resolver"""
        rdf_stub = (
            '<?xml version="1.0"?>'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF>'
        )
        first_path = tmp_path / "first.owl"
        second_path = tmp_path / "second.owl"
        first_path.write_text(rdf_stub)
        second_path.write_text(rdf_stub)

        mock_cognee = MagicMock()
        mock_cognee.cognify = AsyncMock(return_value="success")

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = CognifyCommand()
            args = argparse.Namespace(
                datasets=None,
                chunk_size=None,
                ontology_file=f"{first_path},{second_path}",
                chunker="TextChunker",
                background=False,
                verbose=False,
            )
            command.execute(args)

        config = mock_cognee.cognify.await_args.kwargs["config"]
        resolver = config["ontology_config"]["ontology_resolver"]
        assert resolver.ontology_file == [str(first_path), str(second_path)]

    def test_execute_with_missing_ontology_file(self):
        """A nonexistent --ontology-file fails fast, before any processing"""
        command = CognifyCommand()
        args = argparse.Namespace(
            datasets=None,
            chunk_size=None,
            ontology_file="/nonexistent/ontology.owl",
            chunker="TextChunker",
            background=False,
            verbose=False,
        )

        with pytest.raises(CliCommandException, match="Ontology file not found"):
            command.execute(args)

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.cognify_command.asyncio.run", side_effect=_mock_run)
    def test_cli_call_matches_real_cognify_signature(self, mock_asyncio_run, _mock_resolve):
        """Every kwarg the CLI passes must be an explicit parameter of the real
        cognee.cognify(). Its **kwargs silently swallows stray arguments and
        forwards them to the LLM client, so a mocked call assertion alone
        cannot catch signature drift (issue #3991)."""
        import inspect

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

        passed = set(mock_cognee.cognify.await_args.kwargs)
        real_params = inspect.signature(cognee.cognify).parameters
        explicit = {
            name
            for name, param in real_params.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        unexpected = passed - explicit
        assert not unexpected, (
            f"CLI passes kwargs that are not explicit cognify() parameters: {unexpected}"
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
        assert "all" in actions
        assert "force" in actions

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.delete_command.cognee_datasets")
    @patch("cognee.cli.commands.delete_command.get_datasets_by_name")
    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    @patch("cognee.cli.commands.delete_command.get_deletion_counts")
    @patch("cognee.cli.commands.delete_command.asyncio.run", side_effect=_mock_run)
    def test_execute_delete_dataset_with_confirmation(
        self,
        mock_asyncio_run,
        get_deletion_counts_mock,
        mock_confirm,
        get_datasets_mock,
        datasets_mock,
        _mock_resolve,
    ):
        """Test execute delete dataset with user confirmation"""
        data_directory_path = os.path.join(
            os.path.dirname(__file__), ".data_storage/test_cli_commands"
        )
        cognee_directory_path = os.path.join(
            os.path.dirname(__file__), ".cognee_system/test_cli_commands"
        )

        cognee.config.data_root_directory(data_directory_path)
        cognee.config.system_root_directory(cognee_directory_path)

        asyncio.run(cognee.prune.prune_data())
        asyncio.run(cognee.prune.prune_system(metadata=True))

        asyncio.run(setup())

        expected_dataset_id = uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = expected_dataset_id

        get_datasets_mock.return_value = [mock_dataset]

        get_deletion_counts_mock = AsyncMock()
        get_deletion_counts_mock.return_value = DeletionCountsPreview()

        delete_dataset_mock = AsyncMock()
        datasets_mock.empty_dataset = delete_dataset_mock

        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", all=False, force=False)

        mock_confirm.return_value = True

        command.execute(args)

        delete_dataset_mock.assert_awaited_once_with(dataset_id=expected_dataset_id, user=ANY)

        asyncio.run(cognee.prune.prune_data())
        asyncio.run(cognee.prune.prune_system(metadata=True))

    @patch("cognee.cli.commands.delete_command.get_deletion_counts")
    @patch("cognee.cli.commands.delete_command.fmt.confirm")
    def test_execute_delete_cancelled(self, mock_confirm, mock_get_deletion_counts):
        """Test execute when user cancels deletion"""
        mock_get_deletion_counts = AsyncMock()
        mock_get_deletion_counts.return_value = DeletionCountsPreview()
        command = DeleteCommand()
        args = argparse.Namespace(dataset_name="test_dataset", user_id=None, all=False, force=False)

        mock_confirm.return_value = False

        # Should not raise exception, just return
        command.execute(args)

        mock_confirm.assert_called_once_with(f"Delete dataset '{args.dataset_name}'?")

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.delete_command.cognee_datasets")
    @patch("cognee.cli.commands.delete_command.asyncio.run", side_effect=_mock_run)
    def test_execute_delete_forced(self, mock_asyncio_run, datasets_mock, _mock_resolve):
        """Test execute delete with force flag"""
        delete_all_mock = AsyncMock()
        datasets_mock.delete_all = delete_all_mock

        command = DeleteCommand()
        args = argparse.Namespace(
            dataset_name=None,
            force=True,
            all=True,
        )

        command.execute(args)

        delete_all_mock.assert_awaited_once_with(user=ANY)

    def test_execute_no_delete_target(self):
        """Test execute when no delete target is specified"""
        command = DeleteCommand()
        args = argparse.Namespace(force=True)

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


class TestForgetCommand:
    """Test the ForgetCommand class"""

    def test_command_properties(self):
        command = ForgetCommand()
        assert command.command_string == "forget"
        assert "Remove data" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        command = ForgetCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        actions = {action.dest: action for action in parser._actions}
        assert "dataset" in actions
        assert "dataset_id" in actions
        assert "data_id" in actions
        assert "everything" in actions
        assert "memory_only" in actions
        assert actions["memory_only"].default is False

    @patch("cognee.cli.commands.forget_command.asyncio.run", side_effect=_mock_run)
    def test_execute_threads_memory_only_flag(self, mock_asyncio_run):
        """--memory-only must reach cognee.forget(memory_only=True)."""
        mock_cognee = MagicMock()
        mock_cognee.forget = AsyncMock(
            return_value={"status": "success", "dataset_id": "ds", "data_records_reset": 0}
        )

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = ForgetCommand()
            args = argparse.Namespace(
                dataset="my_dataset",
                dataset_id=None,
                data_id=None,
                everything=False,
                memory_only=True,
            )
            command.execute(args)

        mock_cognee.forget.assert_awaited_once_with(
            data_id=None,
            dataset="my_dataset",
            dataset_id=None,
            everything=False,
            memory_only=True,
        )

    def test_execute_everything_with_memory_only_errors(self):
        """--memory-only has no effect with --everything (which deletes
        outright) -- must error instead of silently doing a full wipe."""
        mock_cognee = MagicMock()
        mock_cognee.forget = AsyncMock()

        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            command = ForgetCommand()
            args = argparse.Namespace(
                dataset=None, dataset_id=None, data_id=None, everything=True, memory_only=True
            )
            # Should not raise, just print an error and return without calling forget().
            command.execute(args)

        mock_cognee.forget.assert_not_awaited()

    def test_execute_no_forget_target(self):
        command = ForgetCommand()
        args = argparse.Namespace(
            dataset=None, dataset_id=None, data_id=None, everything=False, memory_only=False
        )

        # Should not raise, just print an error and return.
        command.execute(args)

    @patch("cognee.cli.commands.forget_command.asyncio.run")
    def test_execute_with_exception(self, mock_asyncio_run):
        mock_asyncio_run.side_effect = Exception("Forget error")

        command = ForgetCommand()
        args = argparse.Namespace(
            dataset="my_dataset",
            dataset_id=None,
            data_id=None,
            everything=False,
            memory_only=False,
        )

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


class TestConfigGetSetPersistence:
    """Exercise the real (unmocked) cognee.config.get/get_all/set behavior.

    These reproduce the originally reported bugs directly against
    cognee.config rather than through ConfigCommand, since that's where the
    actual get/get_all/persistence logic lives.
    """

    def test_get_unknown_key_raises(self):
        from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError

        with pytest.raises(InvalidConfigAttributeError):
            cognee.config.get("not_a_real_config_key")

    def test_get_reflects_in_process_set(self):
        from cognee.infrastructure.data.chunking.config import get_chunk_config

        original = get_chunk_config().chunk_size
        try:
            cognee.config.set("chunk_size", 777)
            assert cognee.config.get("chunk_size") == 777
        finally:
            cognee.config.set_chunk_size(original)

    def test_get_masks_secret_by_default(self):
        from cognee.infrastructure.llm.config import get_llm_config

        original = get_llm_config().llm_api_key
        try:
            cognee.config.set_llm_api_key("sk-1234567890abcdef")

            masked = cognee.config.get("llm_api_key")
            assert masked != "sk-1234567890abcdef"
            assert masked.startswith("sk-")

            full = cognee.config.get("llm_api_key", reveal_secrets=True)
            assert full == "sk-1234567890abcdef"
        finally:
            cognee.config.set_llm_api_key(original)

    def test_get_all_covers_documented_keys(self):
        config_dict = cognee.config.get_all()

        for key in (
            "llm_provider",
            "llm_model",
            "chunk_size",
            "chunk_overlap",
            "vector_db_provider",
            "graph_database_provider",
        ):
            assert key in config_dict

    def test_set_persists_across_process_boundary(self, tmp_path, monkeypatch):
        """Reproduces the originally reported bug: `config set` must survive
        past the current process, since each `cognee-cli` invocation is a
        fresh process re-reading config from scratch."""
        from cognee.infrastructure.data.chunking.config import get_chunk_config

        # A real CHUNK_SIZE env var (e.g. leftover from `dotenv.load_dotenv`
        # picking up a developer's own .env at cognee import time) would
        # outrank the .env file this test writes below, since pydantic-settings
        # prioritizes real environment variables over dotenv-file values.
        monkeypatch.delenv("CHUNK_SIZE", raising=False)

        original_cwd = os.getcwd()
        original_chunk_size = get_chunk_config().chunk_size
        try:
            os.chdir(tmp_path)

            result = cognee.config.set("chunk_size", "999", persist=True)

            assert result["created"] is True
            env_path = tmp_path / ".env"
            assert env_path.exists()
            # dotenv.set_key quotes values, e.g. CHUNK_SIZE='999'.
            assert "CHUNK_SIZE=" in env_path.read_text()
            assert "999" in env_path.read_text()

            # Simulate a fresh process re-reading config from the persisted .env.
            get_chunk_config.cache_clear()
            assert get_chunk_config().chunk_size == 999
        finally:
            os.chdir(original_cwd)
            get_chunk_config.cache_clear()
            get_chunk_config().chunk_size = original_chunk_size


class TestFeedbackCommand:
    """Tests for FeedbackCommand: a missing Q&A entry and a broken cache are
    different failures, and both must exit non-zero."""

    def _add_args(self, **overrides):
        base = dict(feedback_action="add", session_id="s1", qa_id="q1", text="good", score=None)
        base.update(overrides)
        return argparse.Namespace(**base)

    def _delete_args(self):
        return argparse.Namespace(feedback_action="delete", session_id="s1", qa_id="q1")

    def test_command_properties(self):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        command = FeedbackCommand()
        assert command.command_string == "feedback"
        assert "feedback" in command.help_string.lower()

    def test_configure_parser(self):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        parser = argparse.ArgumentParser()
        FeedbackCommand().configure_parser(parser)
        add_args = parser.parse_args(["add", "s1", "q1", "--score", "5"])
        assert add_args.feedback_action == "add"
        assert add_args.score == 5
        del_args = parser.parse_args(["delete", "s1", "q1"])
        assert del_args.feedback_action == "delete"

    def test_add_requires_text_or_score(self):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        with pytest.raises(CliCommandException) as exc_info:
            FeedbackCommand().execute(self._add_args(text=None, score=None))
        assert exc_info.value.error_code == 1

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.feedback_command.fmt.success")
    @patch("cognee.cli.commands.feedback_command.asyncio.run", side_effect=_mock_run)
    def test_add_success(self, _mock_asyncio_run, mock_success, _mock_resolve):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        with patch(
            "cognee.api.v1.session.add_feedback", new_callable=lambda: AsyncMock(return_value=True)
        ) as mock_add:
            FeedbackCommand().execute(self._add_args(score=4))

        mock_add.assert_awaited_once_with(
            session_id="s1", qa_id="q1", feedback_text="good", feedback_score=4, user=ANY
        )
        mock_success.assert_called_once()

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.feedback_command.asyncio.run", side_effect=_mock_run)
    def test_add_not_found_exits_non_zero(self, _mock_asyncio_run, _mock_resolve):
        """False from the SDK means "no such entry" (or caching off), not a crash."""
        from cognee.cli.commands.feedback_command import FeedbackCommand

        with patch(
            "cognee.api.v1.session.add_feedback", new_callable=lambda: AsyncMock(return_value=False)
        ):
            with pytest.raises(CliCommandException) as exc_info:
                FeedbackCommand().execute(self._add_args())

        assert exc_info.value.error_code == 1
        assert "no Q&A entry q1 in session s1" in str(exc_info.value)

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.feedback_command.asyncio.run", side_effect=_mock_run)
    def test_add_infrastructure_error_is_reported_as_error(self, _mock_asyncio_run, _mock_resolve):
        """A cache failure surfaces with its own message, not as "check your IDs"."""
        from cognee.cli.commands.feedback_command import FeedbackCommand
        from cognee.infrastructure.databases.exceptions import CacheConnectionError

        with patch(
            "cognee.api.v1.session.add_feedback",
            new_callable=lambda: AsyncMock(side_effect=CacheConnectionError("redis down")),
        ):
            with pytest.raises(CliCommandException) as exc_info:
                FeedbackCommand().execute(self._add_args())

        assert exc_info.value.error_code == 1
        assert "redis down" in str(exc_info.value)
        assert "no Q&A entry" not in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, CacheConnectionError)

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.feedback_command.fmt.success")
    @patch("cognee.cli.commands.feedback_command.asyncio.run", side_effect=_mock_run)
    def test_delete_success(self, _mock_asyncio_run, mock_success, _mock_resolve):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        with patch(
            "cognee.api.v1.session.delete_feedback",
            new_callable=lambda: AsyncMock(return_value=True),
        ) as mock_delete:
            FeedbackCommand().execute(self._delete_args())

        mock_delete.assert_awaited_once_with(session_id="s1", qa_id="q1", user=ANY)
        mock_success.assert_called_once()

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.feedback_command.asyncio.run", side_effect=_mock_run)
    def test_delete_not_found_exits_non_zero(self, _mock_asyncio_run, _mock_resolve):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        with patch(
            "cognee.api.v1.session.delete_feedback",
            new_callable=lambda: AsyncMock(return_value=False),
        ):
            with pytest.raises(CliCommandException) as exc_info:
                FeedbackCommand().execute(self._delete_args())

        assert exc_info.value.error_code == 1
        assert "no Q&A entry q1 in session s1" in str(exc_info.value)

    @patch(_RESOLVE_USER_PATCH, new_callable=lambda: AsyncMock(return_value=_mock_user()))
    @patch("cognee.cli.commands.feedback_command.asyncio.run", side_effect=_mock_run)
    def test_delete_infrastructure_error_is_reported_as_error(
        self, _mock_asyncio_run, _mock_resolve
    ):
        from cognee.cli.commands.feedback_command import FeedbackCommand

        with patch(
            "cognee.api.v1.session.delete_feedback",
            new_callable=lambda: AsyncMock(side_effect=RuntimeError("cache exploded")),
        ):
            with pytest.raises(CliCommandException) as exc_info:
                FeedbackCommand().execute(self._delete_args())

        assert exc_info.value.error_code == 1
        assert "cache exploded" in str(exc_info.value)
