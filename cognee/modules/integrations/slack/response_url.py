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
from urllib.parse import urlsplit

import aiohttp

logger = logging.getLogger(__name__)

_RESPONSE_URL_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Slack response_urls always live on hooks.slack.com. The value arrives in
# the webhook payload, so without this pin a forged payload could point the
# POST (carrying the answer text) at an attacker-chosen server (SSRF).
_ALLOWED_RESPONSE_URL_HOST = "hooks.slack.com"

# Host alone is not enough. hooks.slack.com also serves Incoming Webhooks under
# /services/..., which anyone can create in a free workspace and which accept the
# exact {"text": ..., "blocks": [...]} body this module sends. Pinning the host but
# not the path still lets a forged payload exfiltrate the answer text into an
# attacker's Slack workspace. Genuine response_urls are only ever these two:
#   slash commands   -> https://hooks.slack.com/commands/T…/…/…
#   interactivity    -> https://hooks.slack.com/actions/T…/…/…
_ALLOWED_RESPONSE_URL_PATH_PREFIXES = ("/commands/", "/actions/")


def _is_slack_response_url(response_url: str) -> bool:
    if not isinstance(response_url, str):
        return False
    try:
        parts = urlsplit(response_url)
    except ValueError:
        return False
    if parts.scheme != "https" or parts.hostname != _ALLOWED_RESPONSE_URL_HOST:
        return False
    return parts.path.startswith(_ALLOWED_RESPONSE_URL_PATH_PREFIXES)


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
    if not _is_slack_response_url(response_url):
        logger.error(
            "Refusing to deliver to a response_url outside https://%s%s",
            _ALLOWED_RESPONSE_URL_HOST,
            "|".join(_ALLOWED_RESPONSE_URL_PATH_PREFIXES),
        )
        return
    try:
        async with aiohttp.ClientSession(timeout=_RESPONSE_URL_TIMEOUT) as session:
            # allow_redirects=False: aiohttp follows redirects by default, and the
            # guard above only ever sees hop 0. A 3xx from hooks.slack.com would
            # otherwise carry this request -- and on 307/308 the answer text itself,
            # since those preserve method and body -- to whatever Location names,
            # including an internal address. Slack never redirects a response_url.
            async with session.post(response_url, json=payload, allow_redirects=False) as response:
                if response.status != 200:
                    logger.warning("Slack response_url POST returned %s", response.status)
    except Exception:  # noqa: BLE001 - nothing left to report to if delivery itself fails
        logger.exception("Failed to deliver a message via Slack response_url")
