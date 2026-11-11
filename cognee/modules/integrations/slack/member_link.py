"""Link a Slack member to a cognee account with Sign in with Slack.

A linked member's ``@cognee`` questions and remembered items use the linked account's
memory (``persistence.resolve_owner_user_id``), so the link has to prove both sides: the
logged-in cognee user starts the flow (the generic integrations ``authorize`` route signs
their id into the state and sets the browser's nonce cookie), and Slack proves which member
signed in. Nothing the member or anyone else types or forwards can bind the two, which is
what the former ``/cognee-link`` code allowed (G42).

It is its own provider, ``slack_member``, so the generic routes give it everything a
connection has: ``POST /api/v1/integrations/slack_member/authorize`` to start, the
state- and nonce-checked callback, a refused link for a member already linked to another
user, the log line of every link, and ``DELETE /api/v1/integrations/slack_member/connection``
to unlink. The Slack app is the same one; Slack asks for the OpenID scopes in a flow of
their own (https://docs.slack.dev/authentication/sign-in-with-slack). Only ``openid`` is
requested: the member's id and workspace, no email or profile.
"""

import hashlib
from typing import Any
from urllib.parse import urlencode

import aiohttp

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.outbound import client_session
from cognee.modules.integrations.slack.handle_slack_link import (
    MEMBER_LINK_PROVIDER,
    member_link_account_id,
)
from cognee.modules.integrations.slack.slack_settings import SlackSettings, require

_AUTHORIZE_URL = "https://slack.com/openid/connect/authorize"
_TOKEN_URL = "https://slack.com/api/openid.connect.token"
_USER_INFO_URL = "https://slack.com/api/openid.connect.userInfo"
_TIMEOUT = aiohttp.ClientTimeout(total=30)

_USER_ID_CLAIM = "https://slack.com/user_id"
_TEAM_ID_CLAIM = "https://slack.com/team_id"


class SlackMemberIntegration(OAuthIntegration):
    provider = MEMBER_LINK_PROVIDER
    settings_cls = SlackSettings

    def authorize_url(self, state: str) -> str:
        query = urlencode(
            {
                "response_type": "code",
                "scope": "openid",
                "client_id": require("client_id"),
                "state": state,
                # Slack asks for a nonce; the identity comes from userInfo over the
                # server-to-server token, not from the ID token, so it only has to be fresh.
                "nonce": hashlib.sha256(state.encode()).hexdigest(),
                "redirect_uri": require("member_redirect_uri"),
            }
        )
        return f"{_AUTHORIZE_URL}?{query}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange the code, then ask Slack who signed in. The user token is not kept."""
        # Neither call follows a redirect: the first body carries the client secret, the
        # second a bearer, and a Location header must not send either elsewhere (SDK-832).
        async with client_session("slack", timeout=_TIMEOUT) as session:
            async with session.post(
                _TOKEN_URL,
                data={
                    "client_id": require("client_id"),
                    "client_secret": require("client_secret"),
                    "code": code,
                    "redirect_uri": require("member_redirect_uri"),
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Slack openid.connect.token failed: HTTP {response.status}")
                token = await response.json()
            if not token.get("ok"):
                raise RuntimeError(
                    f"Slack openid.connect.token failed: {token.get('error', 'unknown')}"
                )
            async with session.post(
                _USER_INFO_URL,
                headers={"Authorization": f"Bearer {token['access_token']}"},
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Slack openid.connect.userInfo failed: HTTP {response.status}"
                    )
                identity = await response.json()
        if not identity.get("ok"):
            raise RuntimeError(
                f"Slack openid.connect.userInfo failed: {identity.get('error', 'unknown')}"
            )
        return identity

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        member = token_response.get(_USER_ID_CLAIM)
        team = token_response.get(_TEAM_ID_CLAIM)
        if not member or not team:
            raise ValueError("Slack userInfo carries no user id or team id")
        return OAuthInstallation(
            provider_account_id=member_link_account_id(team, member),
            token_payload={},
        )

    def state_signing_secret(self) -> str:
        return require("signing_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")
