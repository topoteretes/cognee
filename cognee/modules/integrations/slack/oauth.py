"""Slack OAuth v2 install flow: authorize URL, signed state, code exchange.

https://docs.slack.dev/authentication/installing-with-oauth

The ``state`` parameter is the CSRF defense AND the connecting-user binding:
it is minted only for the authenticated cognee user who initiated the
install, carries ``user_id:expiry`` signed with an HMAC, and the callback
trusts nothing else — the callback itself is necessarily unauthenticated (the
browser arrives from slack.com without a session header), so a valid state is
the only thing that ties the incoming code back to a cognee user.

The signing/validation mechanics themselves are provider-agnostic and live in
:mod:`cognee.modules.integrations.oauth_flow`; this module only supplies
Slack's own signing secret to them, plus everything that IS Slack-specific
(the authorize/token-exchange endpoints, bot scopes, response shape).
"""

from typing import Any, Optional
from uuid import UUID

import aiohttp

from cognee.modules.integrations.oauth_flow import (
    make_state as _make_state,
    sign_state_payload as _sign_state_payload,
    validate_state as _validate_state,
)
from cognee.modules.integrations.slack.slack_settings import require

_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
_ACCESS_URL = "https://slack.com/api/oauth.v2.access"

# Bot scopes: commands, posting, DMs, and channels:read (basic public-channel
# metadata only — name/id/is_private — so the Integrations page can offer a
# per-channel allowlist for slash commands; see slack/channels.py). Still
# deliberately NO channels:history — ingestion isn't built, and the 2025
# non-Marketplace rate limits (1 req/min) make history reads unusable anyway.
_BOT_SCOPES = "commands,chat:write,im:write,channels:read"

# oauth.v2.access is a synchronous call inside the callback request — cap it so
# a hanging Slack never ties up a worker.
_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _sign_state(payload: str) -> str:
    # Keyed with the signing secret — a server-side-only secret Slack also
    # holds, never exposed to the browser or the URL beyond the HMAC itself.
    return _sign_state_payload(payload, signing_secret=require("signing_secret"))


def make_state(user_id: UUID) -> str:
    """Mint the CSRF state binding this install to its initiating cognee user."""
    return _make_state(user_id, signing_secret=require("signing_secret"))


def validate_state(state: str) -> Optional[UUID]:
    """Return the ``user_id`` for a valid, unexpired state; ``None`` otherwise.

    Verifies the HMAC before reading any field, so a forged or tampered state
    never influences behavior — not even error messages.
    """
    return _validate_state(state, signing_secret=require("signing_secret"))


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
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.post(
            _ACCESS_URL,
            data={
                "client_id": require("client_id"),
                "client_secret": require("client_secret"),
                "code": code,
                "redirect_uri": require("redirect_uri"),
            },
        ) as response:
            payload = await response.json()

    if not payload.get("ok"):
        raise RuntimeError(f"Slack oauth.v2.access failed: {payload.get('error', 'unknown')}")

    return payload
