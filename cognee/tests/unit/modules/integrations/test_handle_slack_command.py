"""Unit tests for cognee.modules.integrations.slack.handle_slack_command.

Credential lookup and handle_cognee_ask are patched out. Invariants: an
unconnected workspace is rejected before anything else runs, an empty (or
absent) channel allowlist means unrestricted, and a non-empty allowlist
blocks every channel not on it — regardless of which command was invoked.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest

from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack.handle_slack_command import handle_slack_command

MODULE = "cognee.modules.integrations.slack.handle_slack_command"


def _body(command="/cognee-ask", team_id="T123", channel_id="C1", text="hi"):
    return urlencode(
        {"command": command, "team_id": team_id, "channel_id": channel_id, "text": text}
    ).encode()


def _credential(allowed_channel_ids=None):
    metadata = (
        {"allowed_channel_ids": allowed_channel_ids} if allowed_channel_ids is not None else {}
    )
    return SimpleNamespace(provider_metadata=metadata, status=STATUS_ACTIVE)


@pytest.mark.asyncio
async def test_unconnected_workspace_is_rejected_before_allowlist_check():
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)):
        response = await handle_slack_command(_body())
    assert "not connected" in response["text"]


@pytest.mark.asyncio
async def test_empty_allowlist_permits_every_channel():
    credential = _credential(allowed_channel_ids=[])
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.handle_cognee_ask", new=AsyncMock(return_value={"ok": True})) as ask,
    ):
        response = await handle_slack_command(_body(channel_id="C999"))
    ask.assert_awaited_once()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_absent_allowlist_permits_every_channel():
    credential = _credential(allowed_channel_ids=None)  # key entirely absent
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.handle_cognee_ask", new=AsyncMock(return_value={"ok": True})) as ask,
    ):
        response = await handle_slack_command(_body(channel_id="C999"))
    ask.assert_awaited_once()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_channel_not_on_allowlist_is_blocked():
    credential = _credential(allowed_channel_ids=["C1", "C2"])
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.handle_cognee_ask", new=AsyncMock()) as ask,
    ):
        response = await handle_slack_command(_body(channel_id="C999"))
    ask.assert_not_awaited()
    assert "isn't enabled in this channel" in response["text"]


@pytest.mark.asyncio
async def test_channel_on_allowlist_is_permitted():
    credential = _credential(allowed_channel_ids=["C1", "C2"])
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.handle_cognee_ask", new=AsyncMock(return_value={"ok": True})) as ask,
    ):
        response = await handle_slack_command(_body(channel_id="C2"))
    ask.assert_awaited_once()
    assert response == {"ok": True}


@pytest.mark.asyncio
async def test_cognee_link_dispatches_to_its_handler():
    credential = _credential(allowed_channel_ids=[])
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.handle_cognee_link", new=AsyncMock(return_value={"linked": True})) as link,
    ):
        response = await handle_slack_command(_body(command="/cognee-link", text="some-api-key"))
    link.assert_awaited_once()
    assert response == {"linked": True}


@pytest.mark.asyncio
async def test_unimplemented_command_still_honors_allowlist():
    credential = _credential(allowed_channel_ids=["C1"])
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)):
        response = await handle_slack_command(_body(command="/cognee-forget", channel_id="C999"))
    assert "isn't enabled in this channel" in response["text"]
