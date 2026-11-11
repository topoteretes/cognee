"""Unit tests for cognee.modules.integrations.slack.handle_slack_interactive.

Payload parsing (form-encoded `payload=<json>`, not raw JSON), dispatch to
the "Remember this" message shortcut and the ``/cognee-ask`` answer review
prompt's Share/Discard buttons, and every reply delivered via response_url
rather than the direct HTTP response — matching how Slack actually renders
(or rather, doesn't render) message-action/block-action responses.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.infrastructure.background_tasks import wait_for_background_tasks
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack import handle_cognee_mention as mention
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
)
from cognee.modules.integrations.slack.handle_cognee_mention import NOT_FOLLOWABLE_MESSAGE
from cognee.modules.integrations.slack.handle_slack_interactive import handle_slack_interactive
from cognee.modules.integrations.slack.persistence import CHANNEL_NOT_ENABLED_MESSAGE

MODULE = "cognee.modules.integrations.slack.handle_slack_interactive"
# "Remember this" saves through the shared remember_and_confirm of the @cognee module.
MENTION = "cognee.modules.integrations.slack.handle_cognee_mention"

INVOKING_USER = "U100"


def _credential(user_id=None, allowed_channel_ids=None):
    metadata = {"allowed_channel_ids": allowed_channel_ids} if allowed_channel_ids else {}
    return SimpleNamespace(
        user_id=user_id or uuid4(), status=STATUS_ACTIVE, provider_metadata=metadata
    )


async def _drain_pending_saves():
    """Let the detached save run to completion before asserting on it."""
    await wait_for_background_tasks()


def _body(payload: dict) -> bytes:
    return urlencode({"payload": json.dumps(payload)}).encode()


def _shortcut_payload(
    *,
    team_id="T123",
    channel_id="C1",
    channel_name="general",
    message_text="ship it",
    message_user="U9",
    message_ts="1700000000.000100",
    invoking_user=INVOKING_USER,
    response_url="https://hooks.slack.com/app-actions/T0MJR11A4/21974584944/yk1S9ndf35Q1flupVG5JbpM6",
    callback_id="remember_this",
):
    return {
        "type": "message_action",
        "callback_id": callback_id,
        "team": {"id": team_id},
        "user": {"id": invoking_user},
        "channel": {"id": channel_id, "name": channel_name},
        "message": {"text": message_text, "user": message_user, "ts": message_ts},
        "response_url": response_url,
    }


def _block_action_payload(
    *,
    action_id,
    value="",
    team_id="T123",
    invoking_user=INVOKING_USER,
    response_url="https://hooks.slack.com/app-actions/T0MJR11A4/21974584944/yk1S9ndf35Q1flupVG5JbpM6",
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
    with patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember:
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
        patch(f"{MENTION}.app_follows_channel", new=AsyncMock(return_value=True)),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        response = await handle_slack_interactive(_body(_shortcut_payload()))
        await _drain_pending_saves()

    remember.assert_awaited_once_with(
        credential.user_id,
        text="ship it",
        team_id="T123",
        channel_id="C1",
        message_ts="1700000000.000100",
        channel_name="general",
        author_id="U9",
    )
    post.assert_awaited_once()
    url, reply = post.call_args[0]
    assert (
        url == "https://hooks.slack.com/app-actions/T0MJR11A4/21974584944/yk1S9ndf35Q1flupVG5JbpM6"
    )
    assert "Saved" in reply["text"]
    assert response == {}


@pytest.mark.asyncio
async def test_remember_this_refuses_a_channel_outside_the_allowlist():
    credential = _credential(allowed_channel_ids=["C2"])
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload(channel_id="C1")))

    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert reply["text"] == CHANNEL_NOT_ENABLED_MESSAGE


@pytest.mark.asyncio
async def test_remember_this_saves_from_a_channel_in_the_allowlist():
    credential = _credential(allowed_channel_ids=["C1", "C2"])
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=AsyncMock(return_value=True)),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()),
    ):
        await handle_slack_interactive(_body(_shortcut_payload(channel_id="C1")))
        await _drain_pending_saves()

    remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_remember_this_refuses_a_payload_without_a_channel_once_an_allowlist_is_set():
    credential = _credential(allowed_channel_ids=["C1"])
    payload = _shortcut_payload()
    del payload["channel"]
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()),
    ):
        await handle_slack_interactive(_body(payload))

    remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_remember_this_refuses_a_conversation_the_app_cannot_follow():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=AsyncMock(return_value=False)) as follows,
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload(channel_id="D1")))
        await _drain_pending_saves()

    follows.assert_awaited_once_with(credential, "D1")
    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert reply["text"] == NOT_FOLLOWABLE_MESSAGE
    assert reply["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_remember_this_saves_nothing_when_slack_gives_no_clear_answer():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=AsyncMock(side_effect=RuntimeError("slack"))),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))
        await _drain_pending_saves()

    # An uncertain answer never remembers: the copy could not be kept true.
    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert reply["text"] == mention.SAVE_FAILED_MESSAGE


@pytest.mark.asyncio
async def test_remember_this_answers_slack_before_asking_it_about_the_channel():
    credential = _credential()
    release = asyncio.Event()

    async def slow_answer(*_):
        await release.wait()
        return True

    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=slow_answer),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()),
    ):
        response = await handle_slack_interactive(_body(_shortcut_payload()))
        # Slack's 3-second window is over by now: nothing was saved yet, the work is detached.
        assert response == {}
        remember.assert_not_awaited()
        release.set()
        await _drain_pending_saves()

    remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_detached_save_is_drained_when_the_server_shuts_down():
    from cognee.infrastructure.background_tasks import (
        pending_background_tasks,
        wait_for_background_tasks,
    )

    credential = _credential()
    release = asyncio.Event()

    async def slow_answer(*_):
        await release.wait()
        return True

    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=slow_answer),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()),
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))
        try:
            # The API waits for these on shutdown (wait_for_background_tasks), so a save
            # in flight is not lost to a restart.
            assert pending_background_tasks() == 1
        finally:
            release.set()
            await wait_for_background_tasks()
            await _drain_pending_saves()

    remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_remember_this_rejects_unconnected_workspace():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
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
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))

    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert "@cognee link" in reply["text"]


@pytest.mark.asyncio
async def test_remember_this_rejects_empty_message_text():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.remember_message", new=AsyncMock()) as remember,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload(message_text="  ")))

    remember.assert_not_awaited()
    _, reply = post.call_args[0]
    assert reply["text"] == mention.NOTHING_TO_SAVE_MESSAGE


@pytest.mark.asyncio
async def test_remember_this_reports_deleted_owner():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=AsyncMock(return_value=True)),
        patch(
            f"{MENTION}.remember_message",
            new=AsyncMock(side_effect=EntityNotFoundError(message="gone")),
        ),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))
        await _drain_pending_saves()

    _, reply = post.call_args[0]
    assert reply["text"] == mention.NOT_CONFIGURED_MESSAGE


@pytest.mark.asyncio
async def test_remember_this_reports_unexpected_failure():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=credential.user_id)),
        patch(f"{MENTION}.app_follows_channel", new=AsyncMock(return_value=True)),
        patch(f"{MENTION}.remember_message", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await handle_slack_interactive(_body(_shortcut_payload()))
        await _drain_pending_saves()

    _, reply = post.call_args[0]
    assert reply["text"] == mention.SAVE_FAILED_MESSAGE


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


def _forget_this(**overrides) -> bytes:
    payload = {
        "type": "message_action",
        "callback_id": "forget_this",
        "team": {"id": "T123"},
        "user": {"id": "U100"},
        "channel": {"id": "C1"},
        "message": {"ts": "1700000000.000100", "text": "anything", "user": "U200"},
        "response_url": "https://hooks.slack.com/actions/T123/1/abc",
    }
    payload.update(overrides)
    return urlencode({"payload": json.dumps(payload)}).encode()


@pytest.mark.asyncio
async def test_forget_this_forgets_the_members_own_copy_and_confirms():
    owner = uuid4()
    forget = AsyncMock(return_value=True)
    with (
        patch(
            f"{MODULE}.get_by_team", AsyncMock(return_value=_credential(allowed_channel_ids=["C9"]))
        ),
        patch(f"{MODULE}.resolve_owner_user_id", AsyncMock(return_value=owner)),
        patch(f"{MENTION}.forget_my_remembered_message", forget),
        patch(f"{MODULE}.post_to_response_url", AsyncMock()) as post,
    ):
        assert await handle_slack_interactive(_forget_this()) == {}
        await _drain_pending_saves()

    forget.assert_awaited_once_with(
        owner, team_id="T123", channel_id="C1", message_ts="1700000000.000100"
    )
    assert "Forgotten" in post.await_args.args[1]["text"]


@pytest.mark.asyncio
async def test_forget_this_needs_a_linked_member():
    forget = AsyncMock()
    with (
        patch(f"{MODULE}.get_by_team", AsyncMock(return_value=_credential())),
        patch(f"{MODULE}.resolve_owner_user_id", AsyncMock(return_value=None)),
        patch(f"{MENTION}.forget_my_remembered_message", forget),
        patch(f"{MODULE}.post_to_response_url", AsyncMock()) as post,
    ):
        await handle_slack_interactive(_forget_this())
        await _drain_pending_saves()

    forget.assert_not_awaited()
    assert "@cognee link" in post.await_args.args[1]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("author", "takes_out"), [("U100", True), ("U200", False)], ids=["author", "someone_else"]
)
async def test_forget_this_on_a_synced_reply_takes_it_out_only_for_its_author(author, takes_out):
    """The author re-renders the reply's thread without it; another linked member keeps
    it in the workspace's memory and is told who can take it out (Goran, 2026-11-10)."""
    credential = _credential()
    credential.provider_metadata = {"selected_channels": ["C1"]}
    message = {"ts": "1700000000.000200", "thread_ts": "1700000000.000100", "user": author}
    exclude = AsyncMock(return_value=True)
    with (
        patch(f"{MODULE}.get_by_team", AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", AsyncMock(return_value=uuid4())),
        patch(f"{MENTION}.forget_my_remembered_message", AsyncMock(return_value=False)),
        patch(f"{MENTION}.exclude_message", exclude),
        patch(f"{MODULE}.post_to_response_url", AsyncMock()) as post,
    ):
        await handle_slack_interactive(_forget_this(message=message))
        await _drain_pending_saves()

    if takes_out:
        exclude.assert_awaited_once_with(
            credential, channel_id="C1", message_ts="1700000000.000200", root_ts="1700000000.000100"
        )
        assert "Removed" in post.await_args.args[1]["text"]
    else:
        exclude.assert_not_awaited()
        assert post.await_args.args[1]["text"] == mention.NOT_YOURS_TO_EXCLUDE_MESSAGE
