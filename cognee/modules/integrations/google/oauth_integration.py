"""The OAuth flow every Google connector shares.

Gmail, Drive and Calendar connect one Google account, personal or Workspace, through
an ordinary per-user consent. A provider supplies its scope, settings and dataset
prefix and implements ``list_resources`` and ``sync_now``; the rest lives here.

Two parameters on the authorize URL carry the design:

* ``access_type=offline`` with ``prompt=consent`` is what makes Google issue a refresh
  token. Without both, an account that has consented before gets an access token only,
  and the connector goes dark an hour after install with no way back short of a reconnect.
* ``include_granted_scopes`` is deliberately absent. Incremental authorization would fold
  one product's grant together with another's, and since Google's revoke endpoint kills a
  whole grant, disconnecting one product would silently disconnect the others.

A Workspace-wide install (one service account impersonating every member through
domain-wide delegation) is a different entry point that the generic OAuth callback
cannot express, and is deliberately not attempted here.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar
from urllib.parse import urlencode

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    require_active_credential,
    revoke_credential_if_current,
    update_refreshed_credential,
)
from cognee.modules.integrations.google import client
from cognee.modules.integrations.google.google_settings import GoogleSettings, require
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"

# Refresh this far before the token actually dies, so a sync that takes a
# while does not expire halfway through its own file list.
_REFRESH_MARGIN = timedelta(minutes=5)


class GoogleOAuthIntegration(OAuthIntegration):
    """Subclasses set the class attributes below and add ``list_resources``/``sync_now``."""

    # How the provider reads in log lines and errors, for example "Google Drive".
    display_name: ClassVar[str]
    # The space-separated scope string. ``openid email`` identify the account the
    # credential is keyed on, so every provider asks for them.
    scope: ClassVar[str]
    # The provider's settings instance, not ``settings_cls``: tests and deployments
    # change values on the one object the rest of the code reads.
    settings: ClassVar[GoogleSettings]
    dataset_prefix: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A missing attribute would otherwise surface as an AttributeError in the
        # middle of a consent flow, not when the provider is defined.
        for name in ("display_name", "scope", "settings", "dataset_prefix"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} must set {name}")

    def _require(self, field_name: str) -> str:
        return require(self.settings, field_name, self.settings.model_config["env_prefix"])

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": self._require("client_id"),
            "redirect_uri": self._require("redirect_uri"),
            "response_type": "code",
            "state": state,
            "scope": self.scope,
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{_AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        return await client.exchange_code(
            code,
            client_id=self._require("client_id"),
            client_secret=self._require("client_secret"),
            redirect_uri=self._require("redirect_uri"),
        )

    async def exchange_callback(self, code: str, params: dict[str, str]) -> dict[str, Any]:
        """Exchange the code, then attach the identity of who authorized.

        The token response carries no account identity, but ``parse_installation`` is
        sync and has to derive a stable ``provider_account_id`` from this response
        alone. So the fresh token is spent here, in the async leg, on one userinfo
        call and the result rides along.
        """
        token_response = await self.exchange_code(code)
        userinfo = await client.fetch_userinfo(token_response["access_token"])
        return {**token_response, "userinfo": userinfo}

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        userinfo = token_response.get("userinfo") or {}
        subject = userinfo.get("sub")
        if not subject:
            raise ValueError("Google userinfo response carries no subject")

        token_payload: dict[str, Any] = {"access_token": token_response["access_token"]}
        if token_response.get("refresh_token"):
            token_payload["refresh_token"] = token_response["refresh_token"]
        else:
            # Recoverable only by reconnecting: Google withholds the refresh token when
            # a prior consent is reused, which is what prompt=consent exists to prevent.
            # Better named here than as a 401 an hour from now.
            logger.warning(
                "%s account %s connected without a refresh token; "
                "access will stop working once the access token expires",
                self.display_name,
                subject,
            )

        # hd is present only for Workspace accounts, so its absence is what marks a
        # personal one.
        hosted_domain = userinfo.get("hd")
        return OAuthInstallation(
            provider_account_id=str(subject),
            token_payload=token_payload,
            provider_metadata={
                "email": userinfo.get("email"),
                "hosted_domain": hosted_domain,
                "account_type": "workspace" if hosted_domain else "personal",
                "scope": token_response.get("scope"),
                # A new connection must opt into a selection before importing data.
                # complete_installation preserves existing choices on reconnect.
                self.resource_selection_key: [],
            },
            account_label=userinfo.get("email"),
            scopes=token_response.get("scope"),
            token_expires_at=_expires_at(token_response.get("expires_in")),
        )

    def state_signing_secret(self) -> str:
        return self._require("state_secret")

    def frontend_base_url(self) -> str:
        return self._require("frontend_base_url")

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Connecting updates authorization only; sync starts via the Refresh action."""
        return

    def dataset_name(self, credential: IntegrationCredential) -> str:
        return ingestion.dataset_name_for_credential(self.dataset_prefix, credential)

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Rotate the access token in place.

        An ``invalid_grant`` means the account revoked cognee's access on Google's
        side. Nothing tells us that otherwise, since these connectors receive no
        webhooks, so the local credential is revoked here: the alternative is a
        connection that shows as healthy forever while every sync fails.
        """
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"{self.display_name} credential for account {credential.provider_account_id} "
                "holds no refresh token; the account must reconnect"
            )
        try:
            refreshed = await client.refresh_access_token(
                refresh_token,
                client_id=self._require("client_id"),
                client_secret=self._require("client_secret"),
            )
        except client.GoogleAuthError as error:
            if error.code == "invalid_grant":
                await revoke_credential_if_current(credential)
                logger.warning(
                    "%s access for account %s was revoked at the provider; "
                    "local credential revoked",
                    self.display_name,
                    credential.provider_account_id,
                )
            raise

        await update_refreshed_credential(
            credential,
            # Google returns no new refresh token: the original stays valid
            # until the account revokes it, so it is carried forward.
            token_payload={
                "access_token": refreshed["access_token"],
                "refresh_token": refresh_token,
            },
            scopes=refreshed.get("scope") or credential.scopes,
            token_expires_at=_expires_at(refreshed.get("expires_in")),
        )

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort revoke of this account's grant.

        The grant is kept separate per provider (see the module docstring), so this
        cannot take a sibling Google connector down with it. Still best-effort: the
        local revoke is the actual access cut-off and a network blip must never block
        a disconnect.
        """
        try:
            token_payload = decrypt_token_payload(credential)
            token = token_payload.get("refresh_token") or token_payload.get("access_token")
            if token:
                await client.revoke_token(token)
        except Exception:  # disconnect must not be blocked by Google
            logger.exception(
                "%s token revoke failed for account %s",
                self.display_name,
                credential.provider_account_id,
            )

    async def access_token(self, credential: IntegrationCredential) -> str:
        """Refreshes a token within five minutes of expiry, so callers never see an intermittent 401."""
        credential = await require_active_credential(credential)
        expires_at = credential.token_expires_at
        if expires_at is not None:
            # A naive timestamp comes back from SQLite, which stores no timezone;
            # it is written as UTC, so that is what it is read as.
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at - _REFRESH_MARGIN <= datetime.now(timezone.utc):
                await self.refresh(credential)
                # refresh() writes through its own session, so the instance we
                # were handed still carries the pre-rotation ciphertext. Read the
                # row back rather than decrypting a stale one.
                credential = await require_active_credential(credential)

        token = decrypt_token_payload(credential).get("access_token")
        if not token:
            raise RuntimeError(
                f"{self.display_name} credential for account {credential.provider_account_id} "
                "holds no access token"
            )
        return token


def _expires_at(expires_in: Any) -> datetime | None:
    if not expires_in:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
