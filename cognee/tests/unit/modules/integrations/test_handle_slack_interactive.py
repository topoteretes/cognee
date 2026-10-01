"""Unit tests for cognee.modules.integrations.slack.handle_slack_interactive.

Payload parsing (form-encoded `payload=<json>`, not raw JSON), dispatch to
the "Remember this" message shortcut and the ``/cognee-ask`` answer review
prompt's Share/Discard buttons, and every reply delivered via response_url
rather than the direct HTTP response — matching how Slack actually renders
(or rather, doesn't render) message-action/block-action responses.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
)
from cognee.modules.integrations.slack.handle_slack_interactive import handle_slack_interactive

MODULE = "cognee.modules.integrations.slack.handle_slack_interactive"

INVOKING_USER = "U100"


def _credential(user_id=None):
    return SimpleNamespace(user_id=user_id or uuid4(), status=STATUS_ACTIVE, provider_metadata={})


def _body(payload: dict) -> bytes:
    return urlencode({"payload": json.dumps(payload)}).encode()


def _shortcut_payload(
    *,
    team_id="T123",
    channel_name="general",
    message_text="ship it",
    message_user="U9",
    invoking_user=INVOKING_USER,
    response_url="https://hooks.slack.com/x",
    callback_id="remember_this",
):
    return {
        "type": "message_action",
        "callback_id": callback_id,
        "team": {"id": team_id},
        "user": {"id": invoking_user},
        "channel": {"name": channel_name},
        "message": {"text": message_text, "user": message_user},
        "response_url": response_url,
    }


def _block_action_payload(
    *,
    action_id,
    value="",
    team_id="T123",
    invoking_user=INVOKING_USER,
    response_url="https://hooks.slack.com/x",
):
    return {
        "type": "block_actions",
        "team": {"id": team_id},
        "user": {"id": invoking_user},
        "response_url": response_url,
        "actions": [{"action_id": action_id, "value": value}],
    }


@pytest.mark.asyncio
async def test_missing_payload_field_acks_empty():
    response = await handle_slack_interactive(b"")
    assert response == {}


@pytest.mark.asyncio
async def test_unrecognized_payload_type_acks_empty_without_side_effects():
    payload = {"type": "block_actions", "actions": [{"action_id": "something_else"}]}
    with patch(f"{MODULE}.remember_message", new=AsyncMock()) as remember:
        response = await handle_slack_interactive(_body(payload))

    remember.assert_not_awaited()
    assert response == {}


# ── "Remember this" message shortcut ────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_this_saves_message_and_confirms():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
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
async def test_remember_this_rejects_unauthorized_member():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))

    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert "/cognee-link" in reply["text"]


@pytest.mark.asyncio
async def test_remember_this_rejects_empty_message_text():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
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
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
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
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MODULE}.remember_message", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))

    _, reply = post.call_args[0]
    assert "Could not save" in reply["text"]


# ── /cognee-ask answer review prompt: Share / Discard buttons ──────────────


@pytest.mark.asyncio
async def test_share_button_dispatches_to_handle_cognee_ask_share():
    with (
        patch(
            f"{MODULE}.handle_cognee_ask_share", new=AsyncMock(return_value={"ok": True})
        ) as share,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        response = await handle_slack_interactive(
            _body(
                _block_action_payload(
                    action_id=ASK_SHARE_ACTION_ID,
                    value="abc123",
                    response_url="https://hooks.slack.com/xyz",
                )
            )
        )

    share.assert_awaited_once_with("https://hooks.slack.com/xyz", "abc123")
    post.assert_awaited_once_with("https://hooks.slack.com/xyz", {"ok": True})
    assert response == {}


@pytest.mark.asyncio
async def test_discard_button_dispatches_to_handle_cognee_ask_discard():
    with (
        patch(
            f"{MODULE}.handle_cognee_ask_discard", new=AsyncMock(return_value={"ok": True})
        ) as discard,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        response = await handle_slack_interactive(
            _body(
                _block_action_payload(
                    action_id=ASK_DISCARD_ACTION_ID,
                    value="abc123",
                    response_url="https://hooks.slack.com/xyz",
                )
            )
        )

    discard.assert_awaited_once_with("abc123")
    post.assert_awaited_once_with("https://hooks.slack.com/xyz", {"ok": True})
    assert response == {}


@pytest.mark.asyncio
async def test_unrecognized_block_action_id_is_ignored():
    with (
        patch(f"{MODULE}.handle_cognee_ask_share", new=AsyncMock()) as share,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        response = await handle_slack_interactive(
            _body(_block_action_payload(action_id="some_other_button"))
        )

    share.assert_not_awaited()
    post.assert_not_awaited()
    assert response == {}
