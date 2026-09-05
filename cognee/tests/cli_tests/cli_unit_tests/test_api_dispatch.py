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


class TestImproveDispatch:
    """--api-url improve forwards the same options as the in-process command and
    prints the server's ImproveResult one stage per line."""

    def _args(self, **overrides):
        base = {
            "api_url": "http://localhost:8000",
            "command": "improve",
            "user_id": None,
            "dataset_name": "docs",
            "dataset_id": None,
            "node_name": None,
            "session_ids": ["s1"],
            "background": False,
            "feedback_alpha": 0.2,
            "build_global_context_index": True,
            "build_truth_subspace": True,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def _client(self, MockClient, result):
        mock_instance = MagicMock()
        mock_instance.health.return_value = {"status": "ready"}
        mock_instance.improve.return_value = result
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        return mock_instance

    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_every_option_is_forwarded(self, MockClient):
        client = self._client(MockClient, {"status": "completed", "stages": []})

        dispatch(self._args())

        client.improve.assert_called_once_with(
            dataset_name="docs",
            dataset_id=None,
            node_name=None,
            session_ids=["s1"],
            run_in_background=False,
            build_global_context_index=True,
            build_truth_subspace=True,
            feedback_alpha=0.2,
        )

    @patch("cognee.cli.api_dispatch.fmt")
    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_prints_one_line_per_stage(self, MockClient, mock_fmt):
        self._client(
            MockClient,
            {
                "status": "completed",
                "stages": [
                    {"stage": "feedback_weights", "status": "skipped", "reason": "no_session_ids"},
                    {"stage": "triplet_enrichment", "status": "completed", "counts": {"n": 2}},
                ],
            },
        )

        with patch("cognee.cli.commands.improve_command.fmt") as command_fmt:
            dispatch(self._args())

        command_fmt.success.assert_called_once_with("Knowledge graph improved successfully!")
        printed = [call.args[0] for call in command_fmt.echo.call_args_list]
        assert any("feedback_weights" in line and "no_session_ids" in line for line in printed)
        assert any("triplet_enrichment" in line and "n=2" in line for line in printed)

    @patch("cognee.cli.api_dispatch.fmt")
    @patch("cognee.cli.api_dispatch.CogneeApiClient")
    def test_legacy_server_payload_is_dumped_as_json(self, MockClient, mock_fmt):
        """An older server returns the memify run mapping; keep showing it."""
        self._client(MockClient, {"some-uuid": {"status": "completed"}})

        dispatch(self._args())

        mock_fmt.success.assert_called_once_with("Knowledge graph improved successfully!")
        dumped = [call.args[0] for call in mock_fmt.echo.call_args_list]
        assert any('"some-uuid"' in line for line in dumped)
