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
    def test_health_probe_failure_gives_clear_message(self, MockClient):
        mock_instance = MagicMock()
        mock_instance.health.side_effect = Exception("connection refused")
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        args = argparse.Namespace(
            api_url="http://localhost:9999",
            command="add",
            user_id=None,
        )
        with pytest.raises(RuntimeError, match="Cannot connect to Cognee API"):
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
