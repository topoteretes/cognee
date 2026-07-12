import argparse
import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from cognee.cli.api_dispatch import _dispatch_recall
from cognee.cli.commands.recall_command import RecallCommand


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _args(**overrides):
    values = {
        "query_text": "Where did we leave off?",
        "query_type": None,
        "datasets": None,
        "top_k": 10,
        "system_prompt": None,
        "session_id": "session-1",
        "output_format": "json",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_recall_parser_uses_none_to_represent_no_explicit_type():
    parser = argparse.ArgumentParser()
    RecallCommand().configure_parser(parser)

    args = parser.parse_args(["question", "--session-id", "session-1"])

    assert args.query_type is None


@patch("cognee.cli.user_resolution.resolve_cli_user", new_callable=AsyncMock)
@patch("cognee.cli.commands.recall_command.asyncio.run", side_effect=_run)
@patch("cognee.recall", new_callable=AsyncMock, return_value=[])
def test_unset_type_with_session_uses_session_only_recall(
    mock_recall, _mock_run, mock_resolve_user
):
    user = MagicMock(id="user-1")
    mock_resolve_user.return_value = user

    RecallCommand().execute(_args())

    mock_recall.assert_awaited_once_with(
        query_text="Where did we leave off?",
        top_k=10,
        session_id=ANY,
    )


@patch("cognee.cli.user_resolution.resolve_cli_user", new_callable=AsyncMock)
@patch("cognee.cli.commands.recall_command.asyncio.run", side_effect=_run)
@patch("cognee.recall", new_callable=AsyncMock, return_value=[])
def test_explicit_graph_type_with_session_remains_graph_recall(
    mock_recall, _mock_run, mock_resolve_user
):
    user = MagicMock(id="user-1")
    mock_resolve_user.return_value = user

    RecallCommand().execute(_args(query_type="GRAPH_COMPLETION"))

    assert mock_recall.await_args.kwargs["query_type"].name == "GRAPH_COMPLETION"
    assert mock_recall.await_args.kwargs["session_id"]


def test_api_dispatch_uses_none_for_session_only_and_auto_routed_recall():
    client = MagicMock()
    client.recall.return_value = []

    _dispatch_recall(client, _args())

    client.recall.assert_called_once_with(
        query="Where did we leave off?",
        search_type=None,
        datasets=None,
        top_k=10,
        system_prompt=None,
        session_id="session-1",
    )


def test_api_dispatch_preserves_explicit_hybrid_type():
    client = MagicMock()
    client.recall.return_value = []

    _dispatch_recall(client, _args(query_type="HYBRID_COMPLETION"))

    assert client.recall.call_args.kwargs["search_type"] == "HYBRID_COMPLETION"
