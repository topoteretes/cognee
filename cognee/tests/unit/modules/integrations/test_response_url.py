"""Unit tests for cognee.modules.integrations.slack.response_url.

Invariants: never raises, regardless of a missing URL or a network failure —
callers rely on this to be a safe fire-and-forget delivery.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.integrations.slack.response_url import (
    _normalize_slack_response_url,
    post_to_response_url,
)

# A well-formed response_url. Slack only ever issues /commands/ (slash commands)
# and /actions/ (interactivity); anything else is not a response_url.
_VALID = "https://hooks.slack.com/commands/T00000000/1111111111/abcdefghijklmnop"


@pytest.mark.asyncio
async def test_noop_without_a_url():
    # Must not raise even though there's nowhere to send the payload.
    await post_to_response_url("", {"text": "hi"})


@pytest.mark.asyncio
async def test_never_raises_on_network_failure():
    with patch("aiohttp.ClientSession", side_effect=RuntimeError("network is down")):
        await post_to_response_url(_VALID, {"text": "hi"})  # must not raise


@pytest.mark.parametrize(
    "url",
    [
        _VALID,
        "https://hooks.slack.com/actions/T00000000/1111111111/abcdefghijklmnop",
        "https://HOOKS.SLACK.COM/commands/T0/1/a",
    ],
)
def test_accepts_real_slack_response_urls(url):
    assert _normalize_slack_response_url(url) is not None


@pytest.mark.parametrize(
    ("url", "why"),
    [
        (
            "https://hooks.slack.com/services/T0/B0/tok",
            (
                "Incoming Webhook: same host, and it "
                "accepts the exact body this module sends, so a forged payload would exfiltrate "
                "the answer into an attacker's workspace"
            ),
        ),
        ("https://hooks.slack.com.evil.com/commands/x", "suffix host"),
        ("https://hooks.slack.com@evil.com/commands/x", "userinfo host"),
        ("https://evil@hooks.slack.com/commands/T0/1/a", "userinfo on allowed host"),
        ("http://hooks.slack.com/commands/T0/1/a", "scheme downgrade"),
        ("https://hooks.slack.com/../services/T0/B0/t", "traversal toward /services/"),
        (
            "https://hooks.slack.com/commands/../services/T0/B0/t",
            "traversal after an allowed prefix",
        ),
        (
            "https://hooks.slack.com/commands/%2e%2e/services/T0/B0/t",
            "encoded traversal after an allowed prefix",
        ),
        ("https://hooks.slack.com:444/commands/T0/1/a", "unexpected port"),
        ("https://hooks.slack.com/commands/T0/1/a?next=/services/x", "query string"),
        ("https://hooks.slack.com/commands/T0/1/a#fragment", "fragment"),
        ("https://evil.com/commands/T0/1/a", "right path, wrong host"),
        ("", "empty"),
        (None, "non-string"),
    ],
)
def test_rejects_everything_that_is_not_a_response_url(url, why):
    assert _normalize_slack_response_url(url) is None, why


@pytest.mark.asyncio
async def test_rejected_url_never_opens_an_http_session():
    traversal_url = "https://hooks.slack.com/commands/%2e%2e/services/T0/B0/t"

    with patch("aiohttp.ClientSession") as client_session:
        await post_to_response_url(traversal_url, {"text": "sensitive answer"})

    client_session.assert_not_called()


@pytest.mark.asyncio
async def test_delivery_does_not_follow_redirects():
    """aiohttp follows redirects by default and the host/path guard only sees hop 0,
    so a 3xx from hooks.slack.com would carry the answer text onward -- and 307/308
    preserve method and body."""
    response = MagicMock(status=200)
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = response_context
    supplied_url = _VALID.replace("hooks.slack.com", "HOOKS.SLACK.COM:443")
    with patch("aiohttp.ClientSession", return_value=session):
        await post_to_response_url(supplied_url, {"text": "hi"})

    assert session.post.call_args is not None, "delivery never reached session.post"
    assert session.post.call_args.args[0] == _VALID
    assert session.post.call_args.kwargs.get("allow_redirects") is False
