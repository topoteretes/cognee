"""GitHub (App) as an ``OAuthIntegration`` adapter.

Connects a GitHub org (or user account) through a GitHub App installation
rather than a plain OAuth app: the org admin installs the app once, picks
which repositories it covers, and cognee holds only the installation id —
short-lived access tokens are minted on demand from the app's private key
(:mod:`cognee.modules.integrations.github.app_auth`), so the stored
credential carries an empty token payload.

The install callback differs from the plain-OAuth shape in two ways, both
absorbed by ``exchange_callback``:

* GitHub's redirect carries an ``installation_id`` alongside the ``code``.
* That id arrives on an unauthenticated endpoint and is a small guessable
  integer, so it is never trusted directly: the ``code`` is first exchanged
  for a user token ("Request user authorization during installation" must
  be enabled on the app) and the user's access to the installation is
  confirmed, then the authoritative installation record is fetched with the
  app JWT. Without that check, anyone could bind another org's installation
  — and read access to its private repositories — to their own account.

``revoke_remote`` stays the default no-op on purpose: the GitHub-side
equivalent would be deleting the installation, i.e. uninstalling the app
from the whole org — too destructive for a cognee-side disconnect. Once the
local credential is revoked no further tokens are minted, which is the
actual access cut-off.
"""

import logging
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from cognee.modules.integrations.base import (
    OAuthInstallation,
    OAuthIntegration,
    WebhookVerifier,
)
from cognee.modules.integrations.github import app_auth
from cognee.modules.integrations.github.github_settings import GithubSettings, require
from cognee.modules.integrations.github.handle_github_event import handle_github_event
from cognee.modules.integrations.github.sync import sync_repositories
from cognee.modules.integrations.github.verify_github_signature import GithubWebhookVerifier
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_INSTALL_URL = "https://github.com/apps/{app_slug}/installations/new"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

_TIMEOUT = aiohttp.ClientTimeout(total=30)


class GithubIntegration(OAuthIntegration):
    provider = "github"
    settings_cls = GithubSettings

    def authorize_url(self, state: str) -> str:
        # The app-installation flow, not an OAuth consent screen: GitHub
        # passes ``state`` through to the app's callback URL along with the
        # installation id (and a code, with request-user-authorization on).
        return f"{_INSTALL_URL.format(app_slug=require('app_slug'))}?{urlencode({'state': state})}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange the OAuth code for a *user* token.

        Only used to prove who completed the install flow — the user token
        is checked against the installation and discarded, never stored.
        GitHub returns errors as HTTP 200 with an ``error`` field, so the
        body, not the status, is what's checked.
        """
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(
                _ACCESS_TOKEN_URL,
                data={
                    "client_id": require("client_id"),
                    "client_secret": require("client_secret"),
                    "code": code,
                },
                headers={"Accept": "application/json"},
            ) as response:
                payload = await response.json()

        if payload.get("error") or not payload.get("access_token"):
            raise RuntimeError(
                f"GitHub code exchange failed: {payload.get('error', 'no access_token')}"
            )
        return payload

    async def exchange_callback(self, code: str, params: dict[str, str]) -> dict[str, Any]:
        installation_id = params.get("installation_id", "")
        if not installation_id.isdigit():
            raise ValueError("GitHub callback carried no installation_id")

        user_token_response = await self.exchange_code(code)
        if not await app_auth.user_can_access_installation(
            user_token_response["access_token"], int(installation_id)
        ):
            raise PermissionError(
                f"GitHub user completing the install has no access to "
                f"installation {installation_id}"
            )

        # The authoritative record, fetched as the app — nothing from the
        # unauthenticated query string beyond the (now verified) id is used.
        return await app_auth.get_installation(int(installation_id))

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        installation_id = token_response.get("id")
        if installation_id is None:
            raise ValueError("GitHub installation record carries no id")

        account = token_response.get("account") or {}
        return OAuthInstallation(
            provider_account_id=str(installation_id),
            # No durable per-installation secret exists — tokens are minted
            # on demand from the app private key (see module docstring).
            token_payload={},
            provider_metadata={
                "account_login": account.get("login"),
                "account_id": account.get("id"),
                "account_type": account.get("type"),
                "repository_selection": token_response.get("repository_selection"),
                "app_id": token_response.get("app_id"),
            },
            account_label=account.get("login"),
            auth_type="github_app",
        )

    def state_signing_secret(self) -> str:
        return require("webhook_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    def webhook_verifier(self) -> Optional[WebhookVerifier]:
        return GithubWebhookVerifier()

    async def handle_webhook(self, raw_body: bytes, headers: dict[str, str]) -> None:
        await handle_github_event(raw_body, headers)

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Index every repository the fresh installation covers.

        The ``installation.created`` webhook usually races ahead of the
        OAuth callback storing the credential and is dropped as unknown —
        this hook, firing after the upsert, is what actually performs the
        initial sync.
        """
        await sync_repositories(credential)
