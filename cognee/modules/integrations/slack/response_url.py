"""POST a followup message to a Slack ``response_url``.

Shared by every Slack handler that acks fast and delivers its real answer
later — the async ``/cognee-ask`` search and the "Remember this" message
shortcut both use this. A ``response_url`` is valid for 30 minutes and
accepts up to 5 posts; ``replace_original: true`` (set by the caller's own
payload, not here) swaps an earlier placeholder for the real answer instead
of leaving both visible.
"""

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_RESPONSE_URL_TIMEOUT = aiohttp.ClientTimeout(total=10)


async def post_to_response_url(response_url: str, payload: dict[str, Any]) -> None:
    """Best-effort delivery — never raises.

    There is no caller left to catch anything by the time this typically
    runs (a background task, or after already acking the original webhook),
    so every failure path here ends in a log line instead of a raised
    exception.
    """
    if not response_url:
        logger.error("No response_url to deliver a Slack message to")
        return
    try:
        async with aiohttp.ClientSession(timeout=_RESPONSE_URL_TIMEOUT) as session:
            async with session.post(response_url, json=payload) as response:
                if response.status != 200:
                    logger.warning("Slack response_url POST returned %s", response.status)
    except Exception:  # noqa: BLE001 - nothing left to report to if delivery itself fails
        logger.exception("Failed to deliver a message via Slack response_url")
