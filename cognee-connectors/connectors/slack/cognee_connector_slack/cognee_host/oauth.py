"""Slack OAuth v2 install flow: authorize URL and code exchange.

Copied from cognee.modules.integrations.slack.oauth (cognee dev, 2026-11-12).

https://docs.slack.dev/authentication/installing-with-oauth

The ``state`` parameter (the CSRF defense and the connecting-user binding) is the
integrations router's, signed by :mod:`cognee.modules.integrations.oauth_flow` with a
key only this server holds; this module is what IS Slack-specific (the
authorize/token-exchange endpoints, bot scopes, response shape).
"""

from typing import Any

import aiohttp
from cognee.modules.integrations.outbound import client_session

from cognee_connector_slack.settings import require

_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
_ACCESS_URL = "https://slack.com/api/oauth.v2.access"

# History is fetched only for explicitly selected channels. Metadata scopes
# also let a sync verify both bot and connecting-user membership; users:read renders
# message authors by name; app_mentions:read and im:history receive ``@cognee`` in
# channels and DMs; chat:write answers it with ``chat.postEphemeral``; commands is what
# message shortcuts (Remember this, Forget this) need. Existing installations must
# re-authorize when this list changes. No other write scope: an unused one only widens
# what a leaked bot token could do (SDK-832).
_BOT_SCOPES = (
    "app_mentions:read,chat:write,commands,im:history,channels:read,"
    "channels:history,groups:read,groups:history,users:read"
)

# oauth.v2.access is a synchronous call inside the callback request — cap it so
# a hanging Slack never ties up a worker.
_TIMEOUT = aiohttp.ClientTimeout(total=30)


def build_authorize_url(state: str) -> str:
    """The slack.com/oauth/v2/authorize URL the frontend opens in a popup."""
    return (
        f"{_AUTHORIZE_URL}?client_id={require('client_id')}"
        f"&scope={_BOT_SCOPES}&redirect_uri={require('redirect_uri')}&state={state}"
    )


async def exchange_code(code: str) -> dict[str, Any]:
    """Exchange the OAuth code for tokens at ``oauth.v2.access``.

    Raises ``RuntimeError`` naming Slack's error code when the exchange is
    rejected — Slack returns HTTP 200 with ``ok: false``, so HTTP status
    alone cannot be trusted.
    """
    async with (
        client_session("slack", timeout=_TIMEOUT) as session,
        session.post(
            _ACCESS_URL,
            data={
                "client_id": require("client_id"),
                "client_secret": require("client_secret"),
                "code": code,
                "redirect_uri": require("redirect_uri"),
            },
            # The body carries the client secret: never let a redirect replay
            # it to the host a Location header names.
            allow_redirects=False,
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"Slack oauth.v2.access failed: HTTP {response.status}")
        payload = await response.json()

    if not payload.get("ok"):
        raise RuntimeError(f"Slack oauth.v2.access failed: {payload.get('error', 'unknown')}")

    return payload
