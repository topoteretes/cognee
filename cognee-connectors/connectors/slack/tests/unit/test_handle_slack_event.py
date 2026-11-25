"""Unit tests for cognee_connector_slack.cognee_host.handle_slack_event.

Copied from cognee.tests.unit.modules.integrations.test_handle_slack_event (cognee dev, 2026-11-12).

The dispatch logic only — DB access and the Slack API call are patched out.
Invariants: the url_verification handshake echoes the challenge, both
revoking event types revoke by team_id, app_home_opened publishes a Home
view exactly when there's a connected workspace and a user to publish it
for, and everything (including unknown event types and a broken Home
publish) still acks with 200 so Slack never retry-storms or disables the
endpoint.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from cognee.modules.integrations.credentials import STATUS_ACTIVE

from cognee_connector_slack.cognee_host.handle_slack_event import handle_slack_event

MODULE = "cognee_connector_slack.cognee_host.handle_slack_event"


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


@pytest.mark.asyncio
async def test_url_verification_echoes_challenge():
    response = await handle_slack_event(
        _body({"type": "url_verification", "challenge": "ch4ll3ng3"})
    )
    assert response == {"challenge": "ch4ll3ng3"}


@pytest.mark.asyncio
async def test_app_uninstalled_revokes_by_team():
    envelope = {
        "type": "event_callback",
        "team_id": "T123",
        "event": {"type": "app_uninstalled"},
    }
    with patch(
        "cognee_connector_slack.cognee_host.handle_slack_event.revoke_by_team",
        new=AsyncMock(return_value=True),
    ) as revoke:
        response = await handle_slack_event(_body(envelope))

    revoke.assert_awaited_once_with("T123")
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_tokens_revoked_revokes_by_team():
    # tokens_revoked and app_uninstalled arrive in no guaranteed order — each
    # must independently reach the same end state.
    envelope = {
        "type": "event_callback",
        "team_id": "T123",
        "event": {"type": "tokens_revoked", "tokens": {"bot": ["U1"], "oauth": []}},
    }
    with patch(
        "cognee_connector_slack.cognee_host.handle_slack_event.revoke_by_team",
        new=AsyncMock(return_value=True),
    ) as revoke:
        await handle_slack_event(_body(envelope))

    revoke.assert_awaited_once_with("T123")


@pytest.mark.asyncio
async def test_a_members_revoked_user_token_leaves_the_workspace_connected():
    """Sign in with Slack gives each linked member a user token; Slack reports its
    revocation as ``tokens_revoked`` with ``oauth`` only. The workspace's bot token is
    still live, so the connection stays (G52)."""
    envelope = {
        "type": "event_callback",
        "team_id": "T123",
        "event": {"type": "tokens_revoked", "tokens": {"oauth": ["U0MEMBER"], "bot": []}},
    }
    with patch(
        "cognee_connector_slack.cognee_host.handle_slack_event.revoke_by_team",
        new=AsyncMock(return_value=True),
    ) as revoke:
        await handle_slack_event(_body(envelope))

    revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_unhandled_event_acks_without_revoking():
    envelope = {
        "type": "event_callback",
        "team_id": "T123",
        "event": {"type": "reaction_added"},
    }
    with patch(
        "cognee_connector_slack.cognee_host.handle_slack_event.revoke_by_team",
        new=AsyncMock(),
    ) as revoke:
        response = await handle_slack_event(_body(envelope))

    revoke.assert_not_awaited()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_revoking_event_without_team_id_acks_without_revoking():
    envelope = {"type": "event_callback", "event": {"type": "app_uninstalled"}}
    with patch(
        "cognee_connector_slack.cognee_host.handle_slack_event.revoke_by_team",
        new=AsyncMock(),
    ) as revoke:
        response = await handle_slack_event(_body(envelope))

    revoke.assert_not_awaited()
    assert response == {"ok": True}


# ── app_home_opened ──────────────────────────────────────────────────────────


def _home_envelope(team_id="T123", user="U1"):
    event = {"type": "app_home_opened"}
    if user is not None:
        event["user"] = user
    return {"type": "event_callback", "team_id": team_id, "event": event}


@pytest.mark.asyncio
async def test_app_home_opened_publishes_view_when_connected():
    credential = object()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.is_active", return_value=True),
        patch(f"{MODULE}.decrypt_token_payload", return_value={"access_token": "xoxb-secret"}),
        patch(f"{MODULE}.publish_home_view", new=AsyncMock()) as publish,
    ):
        response = await handle_slack_event(_body(_home_envelope()))

    publish.assert_awaited_once_with("xoxb-secret", "U1")
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_app_home_opened_does_nothing_when_workspace_not_connected():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.is_active", return_value=False),
        patch(f"{MODULE}.publish_home_view", new=AsyncMock()) as publish,
    ):
        response = await handle_slack_event(_body(_home_envelope()))

    publish.assert_not_awaited()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_app_home_opened_without_user_field_does_nothing():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock()) as get_by_team,
        patch(f"{MODULE}.publish_home_view", new=AsyncMock()) as publish,
    ):
        response = await handle_slack_event(_body(_home_envelope(user=None)))

    get_by_team.assert_not_awaited()  # nothing to look up without a user to publish for
    publish.assert_not_awaited()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_app_home_opened_publish_failure_does_not_fail_the_ack():
    credential = object()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.is_active", return_value=True),
        patch(f"{MODULE}.decrypt_token_payload", return_value={"access_token": "xoxb-secret"}),
        patch(
            f"{MODULE}.publish_home_view", new=AsyncMock(side_effect=RuntimeError("not_enabled"))
        ),
    ):
        response = await handle_slack_event(_body(_home_envelope()))

    assert response == {"ok": True}


# ── message events: forget what "Remember this" saved from a message ─────────


def _message_event(**event):
    return {
        "type": "event_callback",
        "team_id": "T123",
        "event": {"type": "message", "channel": "C1", **event},
    }


def _workspace():
    return SimpleNamespace(status=STATUS_ACTIVE, provider_account_id="T123")


@pytest.mark.asyncio
async def test_a_deleted_message_forgets_its_remembered_copies():
    credential = _workspace()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()) as forget,
    ):
        response = await handle_slack_event(
            _body(_message_event(subtype="message_deleted", deleted_ts="1700000000.000100"))
        )

    forget.assert_awaited_once_with("T123", channel_id="C1", message_ts="1700000000.000100")
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_an_edited_message_forgets_its_remembered_copies():
    credential = _workspace()
    edited = {
        "type": "message",
        "ts": "1700000000.000100",
        "text": "owned by Carla now",
        "edited": {"user": "U1", "ts": "1700000500.000000"},
    }
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()) as forget,
    ):
        await handle_slack_event(_body(_message_event(subtype="message_changed", message=edited)))

    # The original message's ts, not the edit's: that is the remembered message.
    forget.assert_awaited_once_with("T123", channel_id="C1", message_ts="1700000000.000100")


@pytest.mark.asyncio
async def test_a_deleted_thread_parent_forgets_its_remembered_copies():
    """A thread parent deleted while its replies stay is not removed from the channel.

    Slack keeps the thread under a "This message was deleted." placeholder: the message
    keeps its ts and replies, its subtype becomes ``tombstone`` (the shape the connector's
    own walk handles), and nobody edited it, so it carries no ``edited``.
    """
    credential = _workspace()
    tombstone = {
        "type": "message",
        "subtype": "tombstone",
        "text": "This message was deleted.",
        "ts": "1700000000.000100",
        "thread_ts": "1700000000.000100",
        "reply_count": 2,
    }
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()) as forget,
    ):
        response = await handle_slack_event(
            _body(_message_event(subtype="message_changed", message=tombstone))
        )

    forget.assert_awaited_once_with("T123", channel_id="C1", message_ts="1700000000.000100")
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_a_change_nobody_made_by_hand_forgets_nothing():
    # message_changed also reports unfurls and language detection; only an edit counts.
    unedited = {"type": "message", "ts": "1700000000.000100", "text": "ship it"}
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock()) as get_by_team,
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()) as forget,
    ):
        response = await handle_slack_event(
            _body(_message_event(subtype="message_changed", message=unedited))
        )

    get_by_team.assert_not_awaited()
    forget.assert_not_awaited()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_a_new_message_is_acked_without_touching_the_database():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock()) as get_by_team,
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()) as forget,
    ):
        response = await handle_slack_event(
            _body(_message_event(text="hello", ts="1700000000.000100", user="U1"))
        )

    get_by_team.assert_not_awaited()
    forget.assert_not_awaited()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_a_deletion_is_followed_after_the_connection_ended():
    """The remembered copies are their members' memory, not the connection's: a workspace
    disconnected without deleting its data still has members whose copies Slack reports."""
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()) as forget,
    ):
        response = await handle_slack_event(
            _body(_message_event(subtype="message_deleted", deleted_ts="1700000000.000100"))
        )

    forget.assert_awaited_once_with("T123", channel_id="C1", message_ts="1700000000.000100")
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_a_failed_forget_reaches_slack_so_it_sends_the_event_again():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=_workspace())),
        patch(
            f"{MODULE}.forget_remembered_message",
            new=AsyncMock(side_effect=RuntimeError("store down")),
        ),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await handle_slack_event(
            _body(_message_event(subtype="message_deleted", deleted_ts="1700000000.000100"))
        )


# --- G6: an edit the root's fingerprint cannot see marks its thread for the next sync ---

ROOT, REPLY = "1700000000.000100", "1700000060.000100"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "root"),
    [
        (
            {
                "subtype": "message_changed",
                "message": {"ts": REPLY, "thread_ts": ROOT, "text": "new", "edited": {"ts": "1"}},
                "previous_message": {"ts": REPLY, "thread_ts": ROOT, "text": "old"},
            },
            ROOT,
        ),
        (
            {
                "subtype": "message_deleted",
                "deleted_ts": REPLY,
                "previous_message": {"ts": REPLY, "thread_ts": ROOT, "text": "old"},
            },
            ROOT,
        ),
        (
            {
                "subtype": "message_changed",
                "message": {"ts": ROOT, "text": "plan", "files": []},
                "previous_message": {"ts": ROOT, "text": "plan", "files": [{"name": "a.pdf"}]},
            },
            ROOT,
        ),
    ],
    ids=["a_reply_edited", "a_reply_deleted", "a_file_removed"],
)
async def test_a_change_the_sync_cannot_see_marks_its_thread(event, root):
    with (
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()),
        patch(f"{MODULE}.mark_thread_changed", new=AsyncMock()) as mark,
    ):
        await handle_slack_event(_body(_message_event(**event)))

    mark.assert_awaited_once_with("T123", "C1", root)


@pytest.mark.asyncio
async def test_a_link_preview_marks_nothing():
    unfurl = {
        "subtype": "message_changed",
        "message": {"ts": REPLY, "thread_ts": ROOT, "text": "see x", "attachments": [{}]},
        "previous_message": {"ts": REPLY, "thread_ts": ROOT, "text": "see x"},
    }
    with (
        patch(f"{MODULE}.forget_remembered_message", new=AsyncMock()),
        patch(f"{MODULE}.mark_thread_changed", new=AsyncMock()) as mark,
    ):
        await handle_slack_event(_body(_message_event(**unfurl)))

    mark.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_uninstall_is_recorded_as_ended_by_the_provider(credential_db):
    from uuid import uuid4

    from cognee.modules.integrations import credentials as store

    from cognee_connector_slack.cognee_host.persistence import revoke_by_team

    owner = uuid4()
    await store.upsert_credential(
        provider="slack",
        provider_account_id="T1",
        user_id=owner,
        token_payload={"access_token": "xoxb"},
    )

    assert await revoke_by_team("T1") is True

    assert (await store.get_ended_connection(owner, "slack")).revoked_by == "provider"


# --- Review of #6683: Slack delivers at least once -----------------------------------------


_QUESTIONS = [
    {"type": "app_mention", "user": "U1", "channel": "C1", "ts": "1.1", "text": "<@B> hi"},
    {"type": "message", "channel_type": "im", "user": "U1", "channel": "D1", "ts": "1.1"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("event", _QUESTIONS, ids=["mention", "direct message"])
async def test_a_question_slack_sends_again_after_a_slow_ack_is_answered_once(event):
    """Slack sends an event again when its ack took longer than 3 seconds
    (``X-Slack-Retry-Reason: http_timeout``): the first delivery arrived and is being answered.
    Answering the retry too runs a second search and posts a second public answer."""
    envelope = {"type": "event_callback", "team_id": "T1", "event_id": "Ev01", "event": event}
    with patch(f"{MODULE}.handle_cognee_mention", new=AsyncMock()) as answer:
        assert await handle_slack_event(_body(envelope)) == {"ok": True}
        assert await handle_slack_event(_body(envelope), retry_reason="http_timeout") == {
            "ok": True
        }
        await asyncio.sleep(0)

    assert answer.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["http_error", "connection_failed"])
async def test_a_question_whose_first_delivery_failed_is_answered_on_the_retry(reason):
    """A retry after a failed delivery is the only time the question reaches the app."""
    envelope = {
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "Ev01",
        "event": _QUESTIONS[0],
    }
    with patch(f"{MODULE}.handle_cognee_mention", new=AsyncMock()) as answer:
        await handle_slack_event(_body(envelope), retry_reason=reason)
        await asyncio.sleep(0)

    assert answer.await_count == 1
