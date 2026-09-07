from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.integrations.slack.history_client import SlackAPIError, SlackHistoryClient
from cognee.modules.integrations.slack.history_models import SlackHistoryError


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self.payload


def client_with(*responses, **kwargs):
    client = SlackHistoryClient("secret-for-test", **kwargs)
    client.session = MagicMock()
    client.session.get.side_effect = responses
    return client


@pytest.mark.asyncio
async def test_history_follows_every_cursor_and_keeps_date_bounds():
    client = client_with(
        Response(
            {"ok": True, "messages": [{"ts": "1"}], "response_metadata": {"next_cursor": "next"}}
        ),
        Response({"ok": True, "messages": [{"ts": "2"}]}),
    )
    assert len(await client.history("C1", oldest="10", latest="20")) == 2
    last = client.session.get.call_args.kwargs
    assert last["params"]["cursor"] == "next"
    assert last["params"]["oldest"] == "10" and last["params"]["latest"] == "20"
    assert last["allow_redirects"] is False
    assert "token" not in last["params"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page",
    [
        {"ok": True, "has_more": True, "messages": []},
        {"ok": True},
    ],
)
async def test_truncated_page_never_succeeds(page):
    client = client_with(Response(page))
    with pytest.raises(SlackHistoryError):
        await client.history("C1", oldest="10", latest="20")


@pytest.mark.asyncio
async def test_repeated_cursor_is_an_error():
    page = {"ok": True, "messages": [], "response_metadata": {"next_cursor": "again"}}
    client = client_with(Response(page), Response(page))
    with pytest.raises(SlackHistoryError, match="repeated"):
        await client.history("C1", oldest="10", latest="20")


@pytest.mark.asyncio
async def test_429_with_non_json_body_respects_retry_after(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("cognee.modules.integrations.slack.history_client.asyncio.sleep", sleep)
    client = client_with(
        Response(None, 429, {"Retry-After": "61"}), Response({"ok": True, "messages": []})
    )
    await client.history("C1", oldest="10", latest="20")
    sleep.assert_awaited_once_with(61)


@pytest.mark.asyncio
async def test_request_cap_fails_before_an_extra_request():
    client = client_with(Response({"ok": True}), max_requests=1)
    await client.call("auth.test")
    with pytest.raises(SlackHistoryError, match="request limit"):
        await client.call("auth.test")
    assert client.session.get.call_count == 1


@pytest.mark.asyncio
async def test_message_cap_is_not_a_successful_partial_import():
    client = client_with(Response({"ok": True, "messages": [{}, {}]}), max_messages=1)
    with pytest.raises(SlackHistoryError, match="message limit"):
        await client.history("C1", oldest="10", latest="20")


@pytest.mark.asyncio
async def test_private_channel_requires_installer_membership_not_only_bot():
    client = client_with(
        Response({"ok": True, "channel": {"id": "G1", "is_member": True, "is_private": True}}),
        Response({"ok": True, "members": ["UOTHER"], "response_metadata": {"next_cursor": "next"}}),
        Response({"ok": True, "members": ["UBOT"]}),
    )
    with pytest.raises(SlackHistoryError, match="connecting Slack user"):
        await client.authorize_channel("G1", "U1")


@pytest.mark.asyncio
async def test_membership_on_later_page_is_accepted():
    client = client_with(
        Response({"ok": True, "channel": {"id": "C1", "is_member": True}}),
        Response({"ok": True, "members": ["UOTHER"], "response_metadata": {"next_cursor": "next"}}),
        Response({"ok": True, "members": ["U1"]}),
    )
    assert (await client.authorize_channel("C1", "U1"))["id"] == "C1"


@pytest.mark.asyncio
async def test_reply_permalink_resolves_parent_and_paginates_replies():
    root, reply = "1700000000.000001", "1701000000.000002"
    client = client_with(
        Response({"ok": True, "messages": [{"ts": reply, "thread_ts": root}]}),
        Response(
            {"ok": True, "messages": [{"ts": root}], "response_metadata": {"next_cursor": "next"}}
        ),
        Response({"ok": True, "messages": [{"ts": root}, {"ts": reply, "thread_ts": root}]}),
    )
    messages = await client.thread("C1", reply)
    assert [message["ts"] for message in messages] == [root, reply]
    assert client.session.get.call_args.kwargs["params"]["ts"] == root


@pytest.mark.asyncio
async def test_missing_scope_is_reported_without_secrets():
    client = client_with(
        Response({"ok": False, "error": "missing_scope", "token": "secret-for-test"})
    )
    with pytest.raises(SlackAPIError, match="missing_scope") as caught:
        await client.call("conversations.history")
    assert "secret-for-test" not in str(caught.value)
