"""Unit tests for cognee.cli.api_dispatch."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from cognee.cli.api_dispatch import can_dispatch, dispatch, SUPPORTED_COMMANDS


class TestCanDispatch:
    def test_false_when_no_api_url(self):
        args = argparse.Namespace(api_url=None, command="add")
        assert can_dispatch(args) is False

    def test_false_when_empty_api_url(self):
        args = argparse.Namespace(api_url="", command="add")
        assert can_dispatch(args) is False

    def test_true_when_api_url_set(self):
        args = argparse.Namespace(api_url="http://localhost:8000", command="add")
        assert can_dispatch(args) is True


class TestDispatchRouting:
    @pytest.mark.parametrize("command", ["cognify", "remember"])
    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_dry_run_rejected_before_remote_request(self, MockClient, command):
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command=command,
            user_id=None,
            dry_run=True,
            data=["test"],
            dataset_name="ds",
            datasets=None,
            background=False,
            chunker="TextChunker",
            chunk_size=None,
            chunks_per_batch=None,
        )

        with pytest.raises(RuntimeError, match="--dry-run is not supported in --api-url mode"):
            dispatch(args)

        MockClient.assert_not_called()

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_unsupported_command_raises(self, MockClient):
        mock_instance = MagicMock()
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="config",
            user_id=None,
        )
        with pytest.raises(RuntimeError, match="not supported in --api-url mode"):
            dispatch(args)

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_unsupported_command_message_suggests_local(self, MockClient):
        mock_instance = MagicMock()
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="sessions",
            user_id=None,
        )
        with pytest.raises(RuntimeError, match="Run without --api-url"):
            dispatch(args)

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_transport_error_gives_clear_message_with_url(self, MockClient):
        """A genuine transport failure on the real request is reported with the
        attempted URL (CLO-321) — no separate /health probe is used."""
        import httpx

        mock_instance = MagicMock()
        mock_instance.add.side_effect = httpx.ConnectError("connection refused")
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:9999",
            command="add",
            user_id=None,
            data=["test"],
            dataset_name="ds",
        )
        with pytest.raises(
            RuntimeError, match="Could not reach the Cognee API at http://localhost:9999"
        ):
            dispatch(args)
        # The old design pre-probed /health; the new one goes straight to the command.
        mock_instance.health.assert_not_called()

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_http_status_error_is_not_masked_as_connection_failure(self, MockClient):
        """A reachable server returning 4xx/5xx must surface its real error, not
        be relabelled 'cannot connect' (the bug behind CLO-321)."""
        mock_instance = MagicMock()
        mock_instance.add.side_effect = RuntimeError("API error 401: unauthorized")
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="https://tenant.aws.cognee.ai",
            command="add",
            user_id=None,
            data=["test"],
            dataset_name="ds",
        )
        with pytest.raises(RuntimeError, match="API error 401"):
            dispatch(args)

    def test_supported_commands_match_dispatchers(self):
        """Ensure SUPPORTED_COMMANDS set matches the dispatcher dict keys."""
        assert SUPPORTED_COMMANDS == {
            "add",
            "cognify",
            "search",
            "memify",
            "datasets",
            "delete",
            "remember",
            "recall",
            "improve",
            "forget",
        }


class TestUserIdHeader:
    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_user_id_passed_as_header(self, MockClient):
        """When --user-id is set, X-User-Id header should be passed to client."""
        mock_instance = MagicMock()
        mock_instance.health.return_value = {"status": "ready"}
        mock_instance.add.return_value = {"status": "ok"}
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="add",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            data=["test"],
            dataset_name="ds",
        )
        dispatch(args)

        # Check that CogneeApiClient was constructed with the X-User-Id header
        call_kwargs = MockClient.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("X-User-Id") == "550e8400-e29b-41d4-a716-446655440000"

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_no_user_id_no_header(self, MockClient):
        mock_instance = MagicMock()
        mock_instance.health.return_value = {"status": "ready"}
        mock_instance.add.return_value = {"status": "ok"}
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="add",
            user_id=None,
            data=["test"],
            dataset_name="ds",
        )
        dispatch(args)

        call_kwargs = MockClient.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "X-User-Id" not in headers


class TestForgetDispatch:
    """Finding 8 (COG-6335 review): --memory-only must reach the API client,
    and a --dataset value must reach it too (args.dataset, not the
    never-set args.dataset_name the dispatcher used to read)."""

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_memory_only_and_dataset_forwarded_to_client(self, MockClient):
        mock_instance = MagicMock()
        mock_instance.forget.return_value = {
            "status": "success",
            "dataset_id": "ds-id",
            "data_records_reset": 0,
        }
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="forget",
            user_id=None,
            dataset="my_dataset",
            dataset_id=None,
            data_id=None,
            everything=False,
            memory_only=True,
        )
        dispatch(args)

        mock_instance.forget.assert_called_once_with(
            dataset="my_dataset",
            dataset_id=None,
            data_id=None,
            everything=False,
            memory_only=True,
        )

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_everything_with_memory_only_does_not_call_client(self, MockClient):
        """--memory-only has no effect with --everything (which deletes
        outright) -- must error instead of silently doing a full wipe."""
        mock_instance = MagicMock()
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="forget",
            user_id=None,
            dataset=None,
            dataset_id=None,
            data_id=None,
            everything=True,
            memory_only=True,
        )
        dispatch(args)

        mock_instance.forget.assert_not_called()

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_dataset_and_dataset_id_both_set_does_not_call_client(self, MockClient):
        mock_instance = MagicMock()
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="forget",
            user_id=None,
            dataset="my_dataset",
            dataset_id="11111111-1111-1111-1111-111111111111",
            data_id=None,
            everything=False,
            memory_only=False,
        )
        dispatch(args)

        mock_instance.forget.assert_not_called()

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_missing_forget_target_does_not_call_client(self, MockClient):
        mock_instance = MagicMock()
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:8000",
            command="forget",
            user_id=None,
            dataset=None,
            dataset_id=None,
            data_id=None,
            everything=False,
            memory_only=False,
        )
        dispatch(args)

        mock_instance.forget.assert_not_called()
