"""Unit tests for cognee.modules.integrations.slack.response_url.

Invariants: never raises, regardless of a missing URL or a network failure —
callers rely on this to be a safe fire-and-forget delivery.
"""

from unittest.mock import patch

import pytest

from cognee.modules.integrations.slack.response_url import post_to_response_url


@pytest.mark.asyncio
async def test_noop_without_a_url():
    # Must not raise even though there's nowhere to send the payload.
    await post_to_response_url("", {"text": "hi"})


@pytest.mark.asyncio
async def test_never_raises_on_network_failure():
    with patch("aiohttp.ClientSession", side_effect=RuntimeError("network is down")):
        await post_to_response_url("https://hooks.slack.com/x", {"text": "hi"})  # must not raise
