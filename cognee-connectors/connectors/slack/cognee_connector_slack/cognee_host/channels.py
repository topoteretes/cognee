"""List a connected Slack workspace's channels for the two pickers.

Copied from cognee.modules.integrations.slack.channels (cognee dev, 2026-11-12).

Backs the channel selection (what is synced) and the command allowlist (where
``@cognee`` and "Remember this" may be used); see
``adapter.visible_channels`` for which names reach either. Requires the
``channels:read`` and ``groups:read`` bot scopes — basic metadata only
(id/name/is_private/is_member), never message content.
"""

from typing import Any

import aiohttp
from cognee.modules.integrations.ingestion import RateLimitedError
from cognee.modules.integrations.outbound import client_session

_CHANNELS_URL = "https://slack.com/api/conversations.list"
_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Slack's own per-page cap for this endpoint.
_PAGE_LIMIT = 200


class SlackChannelListRateLimitedError(RateLimitedError):
    """Slack answered ``ratelimited``; ``retry_after`` is its ``Retry-After``, if any."""

    def __init__(self, retry_after: int | None):
        super().__init__("Slack conversations.list is rate limited")
        self.retry_after = retry_after


async def list_channels(
    access_token: str, *, types: str = "public_channel"
) -> list[dict[str, Any]]:
    """Every channel of ``types`` the bot's token can see, across all pages.

    Returns ``[{"id": ..., "name": ..., "is_private": bool, "is_member": bool}, ...]``,
    where ``is_member`` says whether the app is in the channel. A private channel is
    listed only when the app is a member (``groups:read``). Raises
    ``SlackChannelListRateLimitedError`` on a rate limit, without waiting, and
    ``RuntimeError`` naming Slack's error code on any other rejected call (e.g. the
    workspace connected before ``channels:read`` was added to the app's
    scopes, and hasn't reinstalled since — Slack does not retroactively grant
    new scopes to an existing installation).
    """
    channels: list[dict[str, Any]] = []
    cursor = ""

    async with client_session("slack", timeout=_TIMEOUT) as session:
        while True:
            params = {"types": types, "limit": _PAGE_LIMIT, "exclude_archived": "true"}
            if cursor:
                params["cursor"] = cursor

            payload = await _fetch_page(session, access_token, params)

            channels.extend(
                {
                    "id": c["id"],
                    "name": c["name"],
                    "is_private": c.get("is_private", False),
                    "is_member": c.get("is_member", False),
                    # Slack Connect: shared with another organization.
                    "is_shared": c.get("is_ext_shared", False),
                }
                for c in payload.get("channels", [])
            )

            cursor = (payload.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break

    return channels


async def _fetch_page(
    session: aiohttp.ClientSession, access_token: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Fetch one ``conversations.list`` page.

    The pickers call this inside an HTTP request, so a rate limit is never waited out:
    it raises ``SlackChannelListRateLimitedError`` with Slack's ``Retry-After`` and the
    route tells its client when to try again. Any other rejection raises ``RuntimeError``.
    """
    async with session.get(
        _CHANNELS_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        allow_redirects=False,
    ) as response:
        if 300 <= response.status < 400:
            raise RuntimeError(f"Slack conversations.list failed: HTTP {response.status}")
        payload = await response.json()
        retry_after = response.headers.get("Retry-After")

    if payload.get("ok"):
        return payload
    if payload.get("error") == "ratelimited":
        raise SlackChannelListRateLimitedError(
            int(retry_after) if retry_after and retry_after.isdigit() else None
        )
    raise RuntimeError(f"Slack conversations.list failed: {payload.get('error', 'unknown')}")
