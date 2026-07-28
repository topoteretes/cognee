"""Unit tests for cognee.modules.integrations.slack.home.publish_home_view.

Network is mocked. Invariants: the call is authenticated with the bot token
and targets the given Slack user, and a rejected call raises with Slack's
error code rather than failing silently.
"""

from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.integrations.slack.home import publish_home_view


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


class _FakeSessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


def _session_with(payload):
    session = MagicMock()
    session.post = MagicMock(return_value=_FakeResponse(payload))
    return session


@pytest.mark.asyncio
async def test_publishes_view_for_the_given_user_with_bearer_auth():
    session = _session_with({"ok": True})
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        await publish_home_view("xoxb-secret", "U1")

    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer xoxb-secret"}
    assert kwargs["json"]["user_id"] == "U1"
    assert kwargs["json"]["view"]["type"] == "home"


@pytest.mark.asyncio
async def test_rejected_call_raises_with_slack_error_code():
    session = _session_with({"ok": False, "error": "not_enabled"})
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        with pytest.raises(RuntimeError, match="not_enabled"):
            await publish_home_view("xoxb-secret", "U1")
