"""Unit tests for cognee_connector_slack.cognee_host.channel_membership.

Copied from cognee.tests.unit.modules.integrations.test_slack_channel_membership (cognee dev, 2026-11-12).

"Remember this" saves a message only from a conversation whose edits and deletions
Slack reports to the app: a public or private channel the app is a member of. Direct
messages and group DMs never are, and an unclear answer from Slack raises.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cognee_connector_slack.cognee_host.channel_membership import app_follows_channel
from cognee_connector_slack.source import SlackAPIError, SlackMissingScopeError

MODULE = "cognee_connector_slack.cognee_host.channel_membership"
WORKSPACE = object()


class FakeSlack:
    def __init__(self, channel=None, error=None):
        self.channel = channel
        self.error = error
        self.calls = []

    def call(self, method, **params):
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return {"ok": True, "channel": self.channel}


async def _ask(slack, channel_id="C1"):
    with (
        patch(f"{MODULE}.access_token_for", new=AsyncMock(return_value="xoxb-token")) as token,
        patch(f"{MODULE}.slack_source_module.build_slack_service", return_value=slack),
    ):
        answer = await app_follows_channel(WORKSPACE, channel_id)
    return answer, token


@pytest.mark.asyncio
async def test_a_channel_the_app_is_in_is_followed():
    slack = FakeSlack({"id": "C1", "is_member": True})
    answer, _ = await _ask(slack)

    assert answer is True
    assert slack.calls == [("conversations.info", {"channel": "C1"})]


@pytest.mark.asyncio
async def test_a_private_channel_the_app_is_in_is_followed():
    answer, _ = await _ask(FakeSlack({"id": "G1", "is_private": True, "is_member": True}), "G1")

    assert answer is True


@pytest.mark.asyncio
async def test_a_channel_without_the_app_is_not_followed():
    answer, _ = await _ask(FakeSlack({"id": "C1", "is_member": False}))

    assert answer is False


@pytest.mark.asyncio
async def test_a_private_channel_the_app_cannot_see_is_not_followed():
    answer, _ = await _ask(
        FakeSlack(error=SlackAPIError("conversations.info", "channel_not_found"))
    )

    assert answer is False


@pytest.mark.asyncio
async def test_a_group_dm_is_not_followed():
    answer, _ = await _ask(FakeSlack({"id": "G1", "is_mpim": True, "is_member": True}), "G1")

    assert answer is False


@pytest.mark.asyncio
async def test_a_group_dm_slack_refuses_to_describe_is_not_followed():
    error = SlackMissingScopeError("conversations.info", "mpim:read")
    answer, _ = await _ask(FakeSlack(error=error), "G1")

    assert answer is False


@pytest.mark.asyncio
async def test_a_direct_message_is_not_followed_and_slack_is_not_asked():
    slack = FakeSlack({"id": "D1", "is_im": True})
    answer, token = await _ask(slack, "D1")

    assert answer is False
    assert slack.calls == []
    token.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_answer_about_another_conversation_is_not_followed():
    answer, _ = await _ask(FakeSlack({"id": "C2", "is_member": True}))

    assert answer is False


@pytest.mark.asyncio
async def test_an_unclear_answer_raises():
    with pytest.raises(SlackAPIError):
        await _ask(FakeSlack(error=SlackAPIError("conversations.info", "internal_error")))
