"""Linear (agent app) as an ``OAuthIntegration`` adapter.

Connects a Linear workspace through an *agent* install rather than a plain
OAuth consent: the authorize URL carries ``actor=app``, which installs an
app user — the agent's identity — into the workspace, and the
``app:assignable``/``app:mentionable`` scopes let members delegate issues to
it or @mention it. Those mentions/delegations arrive as agent session
webhooks and are answered from cognee memory
(:mod:`cognee.modules.integrations.linear.agent_session`).

The token exchange differs from the plain-OAuth shape in one way, absorbed
by ``exchange_callback``: Linear's token response says nothing about *which*
workspace authorized the app, but every webhook delivery routes by its
``organizationId`` envelope field — so the fresh token is immediately spent
on one GraphQL query for ``viewer`` (the app user) and ``organization``, and
the merged result is what ``parse_installation`` (which is sync, so it
cannot make that call itself) turns into a credential keyed on the
organization id.

``revoke_remote`` is a real implementation here, unlike GitHub's deliberate
no-op: Linear exposes a cheap token-revoke endpoint that kills only *our*
token, whereas GitHub's remote equivalent would be uninstalling the app from
the whole org — too destructive for a cognee-side disconnect. ``refresh``
stays the base no-op for this first cut (the base contract blesses that);
the refresh token is stored so a later cut can add rotation without a
reconnect.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from cognee.modules.integrations.base import (
    OAuthInstallation,
    OAuthIntegration,
    WebhookVerifier,
)
from cognee.modules.integrations.credentials import decrypt_token_payload
from cognee.modules.integrations.linear.client import graphql
from cognee.modules.integrations.linear.handle_linear_event import handle_linear_event
from cognee.modules.integrations.linear.linear_settings import LinearSettings, require
from cognee.modules.integrations.linear.sync import sync_recent_issues
from cognee.modules.integrations.linear.verify_linear_signature import LinearWebhookVerifier
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_AUTHORIZE_URL = "https://linear.app/oauth/authorize"
_TOKEN_URL = "https://api.linear.app/oauth/token"
_REVOKE_URL = "https://api.linear.app/oauth/revoke"

# app:assignable lets members delegate issues to the agent; app:mentionable
# lets them @mention it. Both require the actor=app install (and actor=app
# cannot be combined with the admin scope).
_SCOPES = "read,write,app:assignable,app:mentionable"

_TIMEOUT = aiohttp.ClientTimeout(total=30)

# viewer is the freshly installed app user (the agent identity in that
# workspace); organization.id is what every webhook envelope routes by.
_INSTALL_CONTEXT_QUERY = """
query InstallContext {
  viewer { id name }
  organization { id name urlKey }
}
"""


def access_token_for(credential: IntegrationCredential) -> str:
    """The credential's decrypted access token — call only at Linear-API time.

    The only sanctioned path from a stored credential to a usable bearer
    token; raises rather than returning an empty string so a malformed
    payload fails loudly at the call site instead of as a Linear 401.
    """
    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise RuntimeError(
            f"Linear credential for organization {credential.provider_account_id} "
            f"holds no access token"
        )
    return token


class LinearIntegration(OAuthIntegration):
    provider = "linear"
    settings_cls = LinearSettings

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": require("client_id"),
            "redirect_uri": require("redirect_uri"),
            "response_type": "code",
            "state": state,
            # The agent install: puts an app user into the workspace instead
            # of acting as the authorizing human.
            "actor": "app",
            "scope": _SCOPES,
        }
        return f"{_AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange the OAuth code for the workspace's agent token."""
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(
                _TOKEN_URL,
                data={
                    "code": code,
                    "redirect_uri": require("redirect_uri"),
                    "client_id": require("client_id"),
                    "client_secret": require("client_secret"),
                    "grant_type": "authorization_code",
                },
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Linear code exchange failed: HTTP {response.status}")
                payload: dict[str, Any] = await response.json()

        if not payload.get("access_token"):
            raise RuntimeError("Linear code exchange returned no access_token")
        return payload

    async def exchange_callback(self, code: str, params: dict[str, str]) -> dict[str, Any]:
        """Exchange the code, then enrich the response with workspace identity.

        The raw token response carries no workspace or app-user identity, but
        ``parse_installation`` is sync and must derive ``provider_account_id``
        from this response alone — and that id must be the organization id,
        because webhooks route back by their ``organizationId`` envelope
        field. So the fresh token is spent here, in the async leg, on one
        GraphQL query and the result rides along.
        """
        token_response = await self.exchange_code(code)
        install_context = await graphql(token_response["access_token"], _INSTALL_CONTEXT_QUERY)
        return {
            **token_response,
            "viewer": install_context.get("viewer"),
            "organization": install_context.get("organization"),
        }

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        organization = token_response.get("organization") or {}
        organization_id = organization.get("id")
        if not organization_id:
            raise ValueError("Linear token response carries no organization id")

        # Secret material stays in token_payload (encrypted at rest); the
        # refresh token is stored unused for now so a later cut can add
        # rotation without forcing a reconnect.
        token_payload = {"access_token": token_response["access_token"]}
        if token_response.get("refresh_token"):
            token_payload["refresh_token"] = token_response["refresh_token"]

        token_expires_at = None
        expires_in = token_response.get("expires_in")
        if expires_in:
            token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

        viewer = token_response.get("viewer") or {}
        return OAuthInstallation(
            provider_account_id=str(organization_id),
            token_payload=token_payload,
            provider_metadata={
                "app_user_id": viewer.get("id"),
                "app_user_name": viewer.get("name"),
                "organization_name": organization.get("name"),
                "organization_url_key": organization.get("urlKey"),
                "scope": token_response.get("scope"),
            },
            account_label=organization.get("name"),
            scopes=token_response.get("scope"),
            token_expires_at=token_expires_at,
            auth_type="oauth2",
        )

    def state_signing_secret(self) -> str:
        return require("webhook_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    def webhook_verifier(self) -> Optional[WebhookVerifier]:
        return LinearWebhookVerifier()

    async def handle_webhook(self, raw_body: bytes, headers: dict[str, str]) -> None:
        await handle_linear_event(raw_body, headers)

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Seed memory with the workspace's recently active issues.

        Issue webhooks only cover changes from now on — and any delivery
        racing ahead of the OAuth callback storing the credential is dropped
        as unknown (same race as GitHub's ``installation.created``). This
        hook, firing after the upsert, is what gives the agent something to
        recall from on day one.
        """
        await sync_recent_issues(credential)

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort remote revoke of the workspace's agent token.

        Cheap and non-destructive on Linear's side (it kills only this
        token, not the app install), so worth doing — but still best-effort:
        the local revoke is the actual access cut-off, and a network blip
        here must never block a disconnect.
        """
        try:
            token = access_token_for(credential)
            async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
                async with session.post(
                    _REVOKE_URL, headers={"Authorization": f"Bearer {token}"}
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            "Linear token revoke for organization %s failed: HTTP %s",
                            credential.provider_account_id,
                            response.status,
                        )
        except Exception:  # noqa: BLE001 - disconnect must proceed no matter what happens here
            logger.exception(
                "Linear token revoke for organization %s failed", credential.provider_account_id
            )
