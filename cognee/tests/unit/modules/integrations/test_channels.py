"""Unit tests for cognee.modules.integrations.slack.channels.list_channels.

Network is mocked — the invariants under test: pagination follows Slack's
next_cursor until exhausted, only the fields the allowlist UI needs are kept,
a rejected call (e.g. missing channels:read) raises rather than returning a
silently-empty list, and a ``ratelimited`` rejection is retried (honoring
``Retry-After``) instead of failing the channel picker on a transient limit.
"""

from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.integrations.slack.channels import list_channels


class _FakeResponse:
    def __init__(self, payload, retry_after=None):
        self._payload = payload
        self.headers = {"Retry-After": retry_after} if retry_after else {}

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
async def test_single_page_returns_id_name_is_private_only():
    session = _session_with_pages(
        [
            {
                "ok": True,
                "channels": [
                    {"id": "C1", "name": "general", "is_private": False, "num_members": 42},
                    {"id": "C2", "name": "eng", "is_private": True},
                ],
                "response_metadata": {"next_cursor": ""},
            }
        ]
    )
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        channels = await list_channels("xoxb-token")

    assert channels == [
        {"id": "C1", "name": "general", "is_private": False},
        {"id": "C2", "name": "eng", "is_private": True},
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
    with patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)):
        with pytest.raises(RuntimeError, match="missing_scope"):
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


# ── ratelimited retry ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ratelimited_retries_and_honors_retry_after_header():
    session = _session_with_responses(
        [
            _FakeResponse({"ok": False, "error": "ratelimited"}, retry_after="0"),
            _FakeResponse(
                {
                    "ok": True,
                    "channels": [{"id": "C1", "name": "general", "is_private": False}],
                    "response_metadata": {"next_cursor": ""},
                }
            ),
        ]
    )
    with (
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
        patch("cognee.modules.integrations.slack.channels.asyncio.sleep", new=_no_sleep),
    ):
        channels = await list_channels("xoxb-token")

    assert channels == [{"id": "C1", "name": "general", "is_private": False}]
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_ratelimited_gives_up_after_max_retries():
    responses = [_FakeResponse({"ok": False, "error": "ratelimited"}) for _ in range(4)]
    session = _session_with_responses(responses)
    with (
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
        patch("cognee.modules.integrations.slack.channels.asyncio.sleep", new=_no_sleep),
    ):
        with pytest.raises(RuntimeError, match="ratelimited"):
            await list_channels("xoxb-token")

    assert session.get.call_count == 4  # 1 initial attempt + 3 retries


async def _no_sleep(_seconds):
    return None
