"""Slack as an ``OAuthIntegration`` adapter.

Wraps the already-tested free functions in :mod:`cognee.modules.integrations
.slack.oauth` behind the generic contract so
:mod:`cognee.api.v1.integrations.routers.get_integrations_router` can
dispatch to Slack the same way it will dispatch to any future provider.
Nothing here changes Slack's own OAuth behavior — it only gives that
behavior a uniform shape a second provider can copy.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.credentials import decrypt_token_payload, upsert_credential
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.slack import oauth as _oauth
from cognee.modules.integrations.slack.slack_settings import SlackSettings, require

logger = logging.getLogger(__name__)

_REVOKE_URL = "https://slack.com/api/auth.revoke"
_TIMEOUT = aiohttp.ClientTimeout(total=30)


class SlackIntegration(OAuthIntegration):
    provider = "slack"
    settings_cls = SlackSettings

    def authorize_url(self, state: str) -> str:
        return _oauth.build_authorize_url(state)

    async def exchange_code(self, code: str) -> dict[str, Any]:
        return await _oauth.exchange_code(code)

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        team = token_response.get("team") or {}
        enterprise = token_response.get("enterprise") or {}

        # Slack's routing id: team.id normally, enterprise.id for a Grid org
        # install.
        account_id = team.get("id") or enterprise.get("id")
        if not account_id:
            raise ValueError("oauth.v2.access response carries neither team.id nor enterprise.id")

        token_payload = {
            "access_token": token_response.get("access_token"),
            "refresh_token": token_response.get("refresh_token"),
        }
        # Non-secret, Slack-specific fields needed later (posting messages,
        # Grid-awareness) — kept out of the encrypted blob so they stay queryable.
        # installed_by_slack_user_id: the Slack user who completed this OAuth
        # install (Slack always populates authed_user.id, even for a bot-only
        # install with no user scopes). Cognee has no real Slack-user <->
        # cognee-user linking yet (tracked separately), so this is used as a
        # stopgap authorization check — see handle_cognee_ask.py and
        # handle_slack_interactive.py's "Remember this" handler — to stop any
        # workspace member from querying/writing to the connecting user's
        # memory, not just the person who actually connected it.
        provider_metadata = {
            "bot_user_id": token_response.get("bot_user_id"),
            "enterprise_id": enterprise.get("id"),
            "installed_by_slack_user_id": (token_response.get("authed_user") or {}).get("id"),
        }

        token_expires_at = None
        expires_in = token_response.get("expires_in")
        if expires_in:
            token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

        return OAuthInstallation(
            provider_account_id=account_id,
            token_payload=token_payload,
            provider_metadata=provider_metadata,
            account_label=team.get("name"),
            scopes=token_response.get("scope"),
            token_expires_at=token_expires_at,
        )

    def state_signing_secret(self) -> str:
        return require("signing_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort ``auth.revoke`` so a disconnected workspace's token
        can't still be used against the Slack API afterwards.

        Never raises — a Slack-side outage or an already-revoked token must
        not block the local disconnect the caller is trying to complete. No
        retry on ``ratelimited``: disconnect doesn't depend on this call
        succeeding, so it's not worth delaying. ``already_revoked`` and
        ``ratelimited`` are logged at info (expected, not actionable);
        anything else is a warning, since that's a real misconfiguration
        (bad credentials, revoked app) worth surfacing in logs.
        """
        token_payload = decrypt_token_payload(credential)
        access_token = token_payload.get("access_token")
        if not access_token:
            return

        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
                async with session.post(
                    _REVOKE_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                ) as response:
                    payload = await response.json()
        except Exception:  # noqa: BLE001 - a failed revoke must never block disconnect
            logger.exception(
                "Slack auth.revoke request failed for account %s", credential.provider_account_id
            )
            return

        error = payload.get("error")
        if not payload.get("ok") and error != "already_revoked":
            log = logger.info if error == "ratelimited" else logger.warning
            log(
                "Slack auth.revoke returned %s for account %s",
                error,
                credential.provider_account_id,
            )

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Rotate the access token via Slack's refresh-token grant.

        A no-op when the stored token payload carries no ``refresh_token`` —
        that's the normal case for most Slack apps, since token rotation is
        an opt-in, irreversible setting on the Slack app itself
        (https://docs.slack.dev/authentication/rotating-and-refreshing-credentials).
        Raises ``RuntimeError`` (naming Slack's error code) if a refresh that
        *should* have worked was rejected — callers decide whether that's
        worth surfacing or just logging.
        """
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            return

        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(
                _oauth._ACCESS_URL,
                data={
                    "client_id": require("client_id"),
                    "client_secret": require("client_secret"),
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            ) as response:
                payload = await response.json()

        if not payload.get("ok"):
            raise RuntimeError(f"Slack token refresh failed: {payload.get('error', 'unknown')}")

        # The refresh response has the same team/enterprise/token shape as
        # the original install response, so the same parsing applies.
        installation = self.parse_installation(payload)
        await upsert_credential(
            provider=self.provider,
            user_id=credential.user_id,
            provider_account_id=installation.provider_account_id,
            token_payload=installation.token_payload,
            account_label=installation.account_label or credential.account_label,
            scopes=installation.scopes or credential.scopes,
            provider_metadata=installation.provider_metadata,
            token_expires_at=installation.token_expires_at,
        )
