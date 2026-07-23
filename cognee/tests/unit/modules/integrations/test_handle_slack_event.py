"""Unit tests for cognee.modules.integrations.slack.handle_slack_event.

The dispatch logic only — DB access and the Slack API call are patched out.
Invariants: the url_verification handshake echoes the challenge, both
revoking event types revoke by team_id, app_home_opened publishes a Home
view exactly when there's a connected workspace and a user to publish it
for, and everything (including unknown event types and a broken Home
publish) still acks with 200 so Slack never retry-storms or disables the
endpoint.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.slack.handle_slack_event import handle_slack_event

MODULE = "cognee.modules.integrations.slack.handle_slack_event"


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
        "cognee.modules.integrations.slack.handle_slack_event.revoke_by_team",
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
        "cognee.modules.integrations.slack.handle_slack_event.revoke_by_team",
        new=AsyncMock(return_value=True),
    ) as revoke:
        await handle_slack_event(_body(envelope))

    revoke.assert_awaited_once_with("T123")


@pytest.mark.asyncio
async def test_unhandled_event_acks_without_revoking():
    envelope = {
        "type": "event_callback",
        "team_id": "T123",
        "event": {"type": "reaction_added"},
    }
    with patch(
        "cognee.modules.integrations.slack.handle_slack_event.revoke_by_team",
        new=AsyncMock(),
    ) as revoke:
        response = await handle_slack_event(_body(envelope))

    revoke.assert_not_awaited()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_revoking_event_without_team_id_acks_without_revoking():
    envelope = {"type": "event_callback", "event": {"type": "app_uninstalled"}}
    with patch(
        "cognee.modules.integrations.slack.handle_slack_event.revoke_by_team",
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
