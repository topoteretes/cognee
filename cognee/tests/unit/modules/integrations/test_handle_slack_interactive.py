"""Unit tests for cognee.modules.integrations.slack.handle_slack_interactive.

Payload parsing (form-encoded `payload=<json>`, not raw JSON), dispatch to
the "Remember this" message shortcut, and every reply is delivered via
response_url rather than the direct HTTP response — matching how Slack
actually renders (or rather, doesn't render) message-action responses.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack.handle_slack_interactive import handle_slack_interactive

MODULE = "cognee.modules.integrations.slack.handle_slack_interactive"


def _credential(user_id=None):
    return SimpleNamespace(user_id=user_id or uuid4(), status=STATUS_ACTIVE)


def _body(payload: dict) -> bytes:
    return urlencode({"payload": json.dumps(payload)}).encode()


def _shortcut_payload(
    *,
    team_id="T123",
    channel_name="general",
    message_text="ship it",
    message_user="U9",
    response_url="https://hooks.slack.com/x",
    callback_id="remember_this",
):
    return {
        "type": "message_action",
        "callback_id": callback_id,
        "team": {"id": team_id},
        "channel": {"name": channel_name},
        "message": {"text": message_text, "user": message_user},
        "response_url": response_url,
    }


@pytest.mark.asyncio
async def test_missing_payload_field_acks_empty():
    response = await handle_slack_interactive(b"")
    assert response == {}


@pytest.mark.asyncio
async def test_unrecognized_payload_type_acks_empty_without_side_effects():
    payload = {"type": "block_actions", "callback_id": "something_else"}
    with patch(f"{MODULE}.remember_message", new=AsyncMock()) as remember:
        response = await handle_slack_interactive(_body(payload))

    remember.assert_not_awaited()
    assert response == {}


@pytest.mark.asyncio
async def test_remember_this_saves_message_and_confirms():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        response = await handle_slack_interactive(_body(_shortcut_payload()))

    remember.assert_awaited_once_with(
        credential.user_id, text="ship it", channel_name="general", author_id="U9"
    )
    post.assert_awaited_once()
    url, reply = post.call_args[0]
    assert url == "https://hooks.slack.com/x"
    assert "Saved" in reply["text"]
    assert response == {}


@pytest.mark.asyncio
async def test_remember_this_rejects_unconnected_workspace():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))

    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert "not connected" in reply["text"]


@pytest.mark.asyncio
async def test_remember_this_rejects_empty_message_text():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload(message_text="  ")))

    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert "Nothing to remember" in reply["text"]


@pytest.mark.asyncio
async def test_remember_this_reports_deleted_owner():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(
            f"{MODULE}.remember_message",
            new=AsyncMock(side_effect=EntityNotFoundError(message="gone")),
        ),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))

    _, reply = post.call_args[0]
    assert "not fully configured" in reply["text"]


@pytest.mark.asyncio
async def test_remember_this_reports_unexpected_failure():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.remember_message", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))

    _, reply = post.call_args[0]
    assert "Could not save" in reply["text"]
