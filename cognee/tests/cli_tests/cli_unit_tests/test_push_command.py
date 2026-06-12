"""
Tests for the push CLI command with proper mocking and coroutine handling.
"""

import argparse
import asyncio
from uuid import uuid4
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from cognee.api.v1.push.push import PushResult
from cognee.cli.commands.push_command import PushCommand
from cognee.cli.exceptions import CliCommandException
from cognee.modules.migration.sources.base import IMPORT_MODES


# Mock asyncio.run to properly handle coroutines
def _mock_run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_user():
    """Return a fake user with an id for resolve_cli_user mocks."""
    user = MagicMock()
    user.id = uuid4()
    return user


def _push_result(dataset_name="main_dataset", target_dataset="main_dataset"):
    return PushResult(
        status="completed",
        dataset_name=dataset_name,
        target_dataset=target_dataset,
        num_nodes=2,
        num_edges=1,
        remote_response={"status": "completed"},
    )


def _fake_resolved_client(service_url="https://resolved.example"):
    """A (client, created) pair as returned by push._resolve_client."""
    client = MagicMock()
    client.service_url = service_url
    client.close = AsyncMock()
    return client, True


class TestPushCommand:
    """Test the PushCommand class"""

    def test_command_properties(self):
        command = PushCommand()
        assert command.command_string == "push"
        assert "Cognee Cloud" in command.help_string
        assert command.docs_url is not None

    def test_configure_parser(self):
        command = PushCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        actions = {action.dest: action for action in parser._actions}
        assert "dataset" in actions
        assert "target_dataset" in actions
        assert "mode" in actions
        assert "url" in actions
        assert "api_key" in actions

        # Defaults
        assert actions["dataset"].default == "main_dataset"
        assert actions["mode"].default == "preserve"

    def test_mode_choices_derived_from_import_modes(self):
        """--mode choices must track IMPORT_MODES, not a hard-coded list (L9)."""
        command = PushCommand()
        parser = argparse.ArgumentParser()

        command.configure_parser(parser)

        actions = {action.dest: action for action in parser._actions}
        assert list(actions["mode"].choices) == list(IMPORT_MODES)
        # A new mode added to IMPORT_MODES is automatically reachable.
        for mode in IMPORT_MODES:
            args = parser.parse_args(["my_dataset", "--mode", mode])
            assert args.mode == mode

    def test_mode_rejects_unknown_value(self):
        command = PushCommand()
        parser = argparse.ArgumentParser()
        command.configure_parser(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["my_dataset", "--mode", "bogus"])

    @patch("cognee.cli.user_resolution.resolve_cli_user", new_callable=AsyncMock)
    @patch("cognee.api.v1.push.push._resolve_client")
    @patch("cognee.push", new_callable=AsyncMock, create=True)
    @patch("cognee.cli.commands.push_command.asyncio.run", side_effect=_mock_run)
    def test_execute_resolves_user_and_passes_it(
        self, mock_asyncio_run, mock_push, mock_resolve_client, mock_resolve_user
    ):
        """--user-id is resolved via resolve_cli_user and passed to cognee.push (M4)."""
        user = _mock_user()
        mock_resolve_user.return_value = user
        mock_resolve_client.return_value = _fake_resolved_client()
        mock_push.return_value = _push_result()

        command = PushCommand()
        args = argparse.Namespace(
            dataset="my_dataset",
            target_dataset=None,
            mode="preserve",
            url=None,
            api_key=None,
            user_id=str(user.id),
        )
        command.execute(args)

        mock_resolve_user.assert_awaited_once_with(str(user.id))
        mock_push.assert_awaited_once_with(
            "my_dataset",
            target_dataset=None,
            mode="preserve",
            url=None,
            api_key=None,
            user=user,
        )

    @patch("cognee.cli.user_resolution.resolve_cli_user", new_callable=AsyncMock)
    @patch("cognee.api.v1.push.push._resolve_client")
    @patch("cognee.push", new_callable=AsyncMock, create=True)
    @patch("cognee.cli.commands.push_command.asyncio.run", side_effect=_mock_run)
    def test_execute_defaults_user_id_to_none(
        self, mock_asyncio_run, mock_push, mock_resolve_client, mock_resolve_user
    ):
        """Without --user-id the resolver is still consulted (default-user path)."""
        mock_resolve_user.return_value = _mock_user()
        mock_resolve_client.return_value = _fake_resolved_client()
        mock_push.return_value = _push_result()

        command = PushCommand()
        # No user_id attribute at all — execute uses getattr(args, "user_id", None).
        args = argparse.Namespace(
            dataset="main_dataset",
            target_dataset=None,
            mode="preserve",
            url=None,
            api_key=None,
        )
        command.execute(args)

        mock_resolve_user.assert_awaited_once_with(None)
        mock_push.assert_awaited_once()

    @patch("cognee.cli.commands.push_command.fmt")
    @patch("cognee.cli.user_resolution.resolve_cli_user", new_callable=AsyncMock)
    @patch("cognee.api.v1.push.push._resolve_client")
    @patch("cognee.push", new_callable=AsyncMock, create=True)
    @patch("cognee.cli.commands.push_command.asyncio.run", side_effect=_mock_run)
    def test_execute_echoes_resolved_url_and_result(
        self, mock_asyncio_run, mock_push, mock_resolve_client, mock_resolve_user, mock_fmt
    ):
        """The resolved remote host is shown before upload; counts come from PushResult (L6)."""
        mock_resolve_user.return_value = _mock_user()
        client, created = _fake_resolved_client("https://resolved.example")
        mock_resolve_client.return_value = (client, created)
        mock_push.return_value = _push_result(target_dataset="remote_name")

        command = PushCommand()
        args = argparse.Namespace(
            dataset="main_dataset",
            target_dataset="remote_name",
            mode="hybrid",
            url=None,
            api_key=None,
            user_id=None,
        )
        command.execute(args)

        echoed = [call.args[0] for call in mock_fmt.echo.call_args_list]
        assert any("https://resolved.example" in message for message in echoed)
        # The throwaway client created for the echo is closed again.
        client.close.assert_awaited_once()

        mock_fmt.success.assert_called_once()
        success_message = mock_fmt.success.call_args.args[0]
        assert "2 nodes" in success_message
        assert "1 edges" in success_message
        assert "remote_name" in success_message

    @patch("cognee.cli.user_resolution.resolve_cli_user", new_callable=AsyncMock)
    @patch("cognee.api.v1.push.push._resolve_client")
    @patch("cognee.push", new_callable=AsyncMock, create=True)
    @patch("cognee.cli.commands.push_command.asyncio.run", side_effect=_mock_run)
    def test_execute_forwards_url_and_api_key(
        self, mock_asyncio_run, mock_push, mock_resolve_client, mock_resolve_user
    ):
        mock_resolve_user.return_value = _mock_user()
        mock_resolve_client.return_value = _fake_resolved_client("https://explicit.example")
        mock_push.return_value = _push_result()

        command = PushCommand()
        args = argparse.Namespace(
            dataset="main_dataset",
            target_dataset=None,
            mode="preserve",
            url="https://explicit.example",
            api_key="ck_test",
            user_id=None,
        )
        command.execute(args)

        mock_resolve_client.assert_called_once_with("https://explicit.example", "ck_test")
        push_kwargs = mock_push.await_args.kwargs
        assert push_kwargs["url"] == "https://explicit.example"
        assert push_kwargs["api_key"] == "ck_test"

    @patch("cognee.cli.commands.push_command.asyncio.run")
    def test_execute_with_exception(self, mock_asyncio_run):
        """Test execute handles exceptions properly"""
        command = PushCommand()
        args = argparse.Namespace(
            dataset="main_dataset",
            target_dataset=None,
            mode="preserve",
            url=None,
            api_key=None,
            user_id=None,
        )

        def fail(coro):
            coro.close()  # avoid "coroutine never awaited" warnings
            raise Exception("Push error")

        mock_asyncio_run.side_effect = fail

        with pytest.raises(CliCommandException):
            command.execute(args)
