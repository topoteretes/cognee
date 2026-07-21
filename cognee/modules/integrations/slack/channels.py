"""List a connected Slack workspace's public channels.

Backs the per-channel allowlist on the Integrations page: a workspace owner
picks which channels ``/cognee-ask`` (and future slash commands) may be used
in, rather than the whole workspace by default. Requires the ``channels:read``
bot scope — basic metadata only (id/name/is_private), never message content.
"""

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_CHANNELS_URL = "https://slack.com/api/conversations.list"
_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Slack's own per-page cap for this endpoint.
_PAGE_LIMIT = 200

# This call runs synchronously inside the channel-picker's GET request, so a
# single transient rate limit shouldn't hard-fail the settings page.
_MAX_RATELIMIT_RETRIES = 3
_DEFAULT_RETRY_AFTER_SECONDS = 1


async def list_channels(access_token: str) -> list[dict[str, Any]]:
    """Every public channel the bot's token can see, across all pages.

    Returns ``[{"id": ..., "name": ..., "is_private": bool}, ...]``. Raises
    ``RuntimeError`` naming Slack's error code on a rejected call (e.g. the
    workspace connected before ``channels:read`` was added to the app's
    scopes, and hasn't reinstalled since — Slack does not retroactively grant
    new scopes to an existing installation).
    """
    channels: list[dict[str, Any]] = []
    cursor = ""

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        while True:
            params = {"types": "public_channel", "limit": _PAGE_LIMIT, "exclude_archived": "true"}
            if cursor:
                params["cursor"] = cursor

            payload = await _fetch_page(session, access_token, params)

            channels.extend(
                {"id": c["id"], "name": c["name"], "is_private": c.get("is_private", False)}
                for c in payload.get("channels", [])
            )

            cursor = (payload.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break

    return channels


async def _fetch_page(
    session: aiohttp.ClientSession, access_token: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Fetch one ``conversations.list`` page, retrying on Slack's rate limit.

    A rate-limited call comes back as HTTP 429 with a ``Retry-After`` header
    (seconds) and ``{"ok": false, "error": "ratelimited"}`` in the body —
    retried up to ``_MAX_RATELIMIT_RETRIES`` times, honoring that header,
    before giving up. Any other rejection raises immediately.
    """
    attempt = 0
    while True:
        async with session.get(
            _CHANNELS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        ) as response:
            payload = await response.json()
            retry_after = response.headers.get("Retry-After")

        if payload.get("ok"):
            return payload

        if payload.get("error") != "ratelimited" or attempt >= _MAX_RATELIMIT_RETRIES:
            raise RuntimeError(f"Slack conversations.list failed: {payload.get('error', 'unknown')}")

        delay = int(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_AFTER_SECONDS
        attempt += 1
        logger.warning(
            "Slack conversations.list rate-limited, retrying in %ss (attempt %s/%s)",
            delay,
            attempt,
            _MAX_RATELIMIT_RETRIES,
        )
        await asyncio.sleep(delay)
