"""Unit tests for cognee.modules.integrations.slack.channels.list_channels.

Network is mocked — the invariants under test: pagination follows Slack's
next_cursor until exhausted, only the fields the allowlist UI needs are kept,
a rejected call (e.g. missing channels:read) raises rather than returning a
silently-empty list, and a ``ratelimited`` rejection raises at once with Slack's
``Retry-After``: the pickers answer an HTTP request, so a rate limit is never waited out.
"""

from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.integrations.ingestion import RateLimitedError
from cognee.modules.integrations.slack.channels import list_channels


class _FakeResponse:
    def __init__(self, payload, retry_after=None):
        self._payload = payload
        self.headers = {"Retry-After": retry_after} if retry_after else {}
        # Slack answers a rate limit with 429 and everything else (``ok: false``
        # included) with 200.
        self.status = 429 if retry_after else 200

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


def _session_with_pages(pages):
    session = MagicMock()
    responses = [_FakeResponse(page) for page in pages]
    session.get = MagicMock(side_effect=responses)
    return session


def _session_with_responses(responses):
    session = MagicMock()
    session.get = MagicMock(side_effect=responses)
    return session


@pytest.mark.asyncio
async def test_single_page_returns_id_name_is_private_and_is_member_only():
    session = _session_with_pages(
        [
            {
                "ok": True,
                "channels": [
                    {
                        "id": "C1",
                        "name": "general",
                        "is_private": False,
                        "is_member": True,
                        "num_members": 42,
                    },
                    {"id": "C2", "name": "eng", "is_private": True},
                    {"id": "C3", "name": "acme", "is_ext_shared": True, "is_member": True},
                ],
                "response_metadata": {"next_cursor": ""},
            }
        ]
    )
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        channels = await list_channels("xoxb-token")

    assert channels == [
        {"id": "C1", "name": "general", "is_private": False, "is_member": True, "is_shared": False},
        {"id": "C2", "name": "eng", "is_private": True, "is_member": False, "is_shared": False},
        {"id": "C3", "name": "acme", "is_private": False, "is_member": True, "is_shared": True},
    ]


@pytest.mark.asyncio
async def test_follows_pagination_cursor_across_multiple_pages():
    session = _session_with_pages(
        [
            {
                "ok": True,
                "channels": [{"id": "C1", "name": "general", "is_private": False}],
                "response_metadata": {"next_cursor": "page2"},
            },
            {
                "ok": True,
                "channels": [{"id": "C2", "name": "eng", "is_private": False}],
                "response_metadata": {"next_cursor": ""},
            },
        ]
    )
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        channels = await list_channels("xoxb-token")

    assert [c["id"] for c in channels] == ["C1", "C2"]
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_rejected_call_raises_with_slack_error_code():
    session = _session_with_pages([{"ok": False, "error": "missing_scope"}])
    with (
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
        pytest.raises(RuntimeError, match="missing_scope"),
    ):
        await list_channels("xoxb-token")


@pytest.mark.asyncio
async def test_sends_bearer_token_and_public_channel_filter():
    session = _session_with_pages(
        [{"ok": True, "channels": [], "response_metadata": {"next_cursor": ""}}]
    )
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        await list_channels("xoxb-secret")

    _, kwargs = session.get.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer xoxb-secret"}
    assert kwargs["params"]["types"] == "public_channel"


@pytest.mark.asyncio
async def test_the_channel_types_can_include_private_channels():
    session = _session_with_pages(
        [{"ok": True, "channels": [], "response_metadata": {"next_cursor": ""}}]
    )
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        await list_channels("xoxb-secret", types="public_channel,private_channel")

    assert session.get.call_args.kwargs["params"]["types"] == "public_channel,private_channel"


# ── ratelimited ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_rate_limited_page_raises_at_once_with_its_retry_after():
    """No wait inside the request: the client is told when to try again."""
    session = _session_with_responses(
        [
            _FakeResponse({"ok": False, "error": "ratelimited"}, retry_after="30"),
            _FakeResponse(
                {
                    "ok": True,
                    "channels": [{"id": "C1", "name": "general", "is_private": False}],
                    "response_metadata": {"next_cursor": ""},
                }
            ),
        ]
    )
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    with (
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
        patch("asyncio.sleep", new=sleep),
        pytest.raises(RateLimitedError) as caught,
    ):
        await list_channels("xoxb-token")

    assert caught.value.retry_after == 30
    assert slept == []
    assert session.get.call_count == 1
