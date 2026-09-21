"""Google Drive as an ``OAuthIntegration`` adapter.

Connects one Google account, personal or Workspace, through an ordinary
per-user OAuth consent. The two are the same code path: they differ only by
the ``hd`` claim the userinfo endpoint reports, which is recorded as
``account_type`` exactly the way the GitHub adapter records ``User`` versus
``Organization`` for an install. A Workspace-wide install (one service
account impersonating every member through domain-wide delegation) is a
different entry point that the generic OAuth callback cannot express at all,
and is deliberately not attempted here.

Two parameters on the authorize URL carry the design:

* ``access_type=offline`` with ``prompt=consent`` is what makes Google issue
  a refresh token. Without both, an account that has consented before gets an
  access token only, and this connector goes dark an hour after install with
  no way back short of a reconnect.
* ``include_granted_scopes`` is deliberately **absent**. Incremental
  authorization would fold Drive's grant together with any later Gmail or
  Calendar grant, and since Google's revoke endpoint kills a whole grant,
  disconnecting one product would silently disconnect the others.

``refresh`` is a real implementation here, unlike Slack's and unlike the
GitHub adapter's no-op: Google access tokens last about an hour, so a
connector that cannot rotate is a connector that works until lunch.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    get_credential_by_account,
    revoke_credential_by_account,
    upsert_credential,
)
from cognee.modules.integrations.google_drive import client
from cognee.modules.integrations.google_drive.google_drive_settings import (
    GoogleDriveSettings,
    require,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"

# drive.readonly is a restricted scope and needs Google's OAuth verification
# before a public launch; openid and email are what identify the account the
# credential is keyed on.
_SCOPES = "openid email https://www.googleapis.com/auth/drive.readonly"

# Refresh this far before the token actually dies, so a sync that takes a
# while does not expire halfway through its own file list.
_REFRESH_MARGIN = timedelta(minutes=5)


class GoogleDriveIntegration(OAuthIntegration):
    provider = "google_drive"
    settings_cls = GoogleDriveSettings

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": require("client_id"),
            "redirect_uri": require("redirect_uri"),
            "response_type": "code",
            "state": state,
            "scope": _SCOPES,
            # See the module docstring: both are required for a refresh token,
            # and include_granted_scopes is deliberately not among them.
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{_AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        return await client.exchange_code(
            code,
            client_id=require("client_id"),
            client_secret=require("client_secret"),
            redirect_uri=require("redirect_uri"),
        )

    async def exchange_callback(self, code: str, params: dict[str, str]) -> dict[str, Any]:
        """Exchange the code, then attach the identity of who authorized.

        The token response carries no account identity, but
        ``parse_installation`` is sync and has to derive a stable
        ``provider_account_id`` from this response alone. So the fresh token
        is spent here, in the async leg, on one userinfo call and the result
        rides along. Same shape as the Linear adapter, for the same reason.
        """
        token_response = await self.exchange_code(code)
        userinfo = await client.fetch_userinfo(token_response["access_token"])
        return {**token_response, "userinfo": userinfo}

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        userinfo = token_response.get("userinfo") or {}
        subject = userinfo.get("sub")
        if not subject:
            raise ValueError("Google userinfo response carries no subject")

        token_payload = {"access_token": token_response["access_token"]}
        refresh_token = token_response.get("refresh_token")
        if refresh_token:
            token_payload["refresh_token"] = refresh_token
        else:
            # Recoverable, but only by reconnecting: Google withholds the
            # refresh token when a prior consent is reused, which is what
            # prompt=consent exists to prevent. Worth naming loudly here
            # rather than as a 401 an hour from now.
            logger.warning(
                "Google Drive account %s connected without a refresh token; "
                "access will stop working once the access token expires",
                subject,
            )

        # hd is present only for Workspace accounts, so its absence is what
        # marks a personal one. Mirrors GitHub's account_type.
        hosted_domain = userinfo.get("hd")
        return OAuthInstallation(
            provider_account_id=str(subject),
            token_payload=token_payload,
            provider_metadata={
                "email": userinfo.get("email"),
                "hosted_domain": hosted_domain,
                "account_type": "workspace" if hosted_domain else "personal",
                "scope": token_response.get("scope"),
            },
            account_label=userinfo.get("email"),
            scopes=token_response.get("scope"),
            token_expires_at=_expires_at(token_response.get("expires_in")),
            auth_type="oauth2",
        )

    def state_signing_secret(self) -> str:
        return require("state_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Index the account's Drive once, right after connecting.

        Google offers no signed webhook on this path and cognee runs no
        durable scheduler, so this initial pass is what the account's memory
        is built from until someone asks for a re-sync.
        """
        from cognee.modules.integrations.google_drive.sync import sync_drive

        await sync_drive(credential)

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Rotate the access token in place.

        An ``invalid_grant`` means the account revoked cognee's access on
        Google's side. Nothing tells us that otherwise, since this connector
        receives no webhooks, so the local credential is revoked here: the
        alternative is a connection that shows as healthy forever while every
        sync fails.
        """
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"Google Drive credential for account {credential.provider_account_id} "
                f"holds no refresh token; the account must reconnect"
            )

        try:
            refreshed = await client.refresh_access_token(
                refresh_token,
                client_id=require("client_id"),
                client_secret=require("client_secret"),
            )
        except client.GoogleAuthError as error:
            if error.code == "invalid_grant":
                await revoke_credential_by_account(self.provider, credential.provider_account_id)
                logger.warning(
                    "Google Drive access for account %s was revoked at the provider; "
                    "local credential revoked",
                    credential.provider_account_id,
                )
            raise

        await upsert_credential(
            provider=self.provider,
            user_id=credential.user_id,
            provider_account_id=credential.provider_account_id,
            # Google returns no new refresh token: the original stays valid
            # until the account revokes it, so it is carried forward.
            token_payload={
                "access_token": refreshed["access_token"],
                "refresh_token": refresh_token,
            },
            token_expires_at=_expires_at(refreshed.get("expires_in")),
            # Every one of these is assigned unconditionally by
            # upsert_credential, so anything left out is cleared, on a path
            # that runs hourly. provider_metadata matters most: the account's
            # email lives there and the dataset name is derived from it, so
            # dropping it would send the next sync to a different dataset and
            # split the account's memory in two.
            account_label=credential.account_label,
            auth_type=credential.auth_type,
            scopes=refreshed.get("scope") or credential.scopes,
            provider_metadata=credential.provider_metadata,
            workspace_id=credential.workspace_id,
        )

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort revoke of this account's Drive grant.

        Cheap and correctly scoped because the grant was kept separate (see
        the module docstring), so this cannot take a sibling Google connector
        down with it. Still best-effort: the local revoke is the actual
        access cut-off and a network blip must never block a disconnect.
        """
        try:
            token_payload = decrypt_token_payload(credential)
            token = token_payload.get("refresh_token") or token_payload.get("access_token")
            if token:
                await client.revoke_token(token)
        except Exception:  # disconnect must proceed no matter what happens here
            logger.exception(
                "Google Drive token revoke for account %s failed",
                credential.provider_account_id,
            )


def _expires_at(expires_in: Any) -> datetime | None:
    if not expires_in:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))


async def access_token_for(credential: IntegrationCredential) -> str:
    """A usable access token for this credential, refreshed if it is about to die.

    The only sanctioned path from a stored credential to a bearer token.
    Refreshing is transparent to callers because every caller would otherwise
    have to reimplement the same expiry check, and getting it wrong shows up
    as an intermittent 401 rather than an error anyone can act on.
    """
    expires_at = credential.token_expires_at
    if expires_at is not None:
        # A naive timestamp comes back from SQLite, which stores no timezone;
        # it is written as UTC, so that is what it is read as.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at - _REFRESH_MARGIN <= datetime.now(timezone.utc):
            await GoogleDriveIntegration().refresh(credential)
            # refresh() writes through its own session, so the instance we
            # were handed still carries the pre-rotation ciphertext. Read the
            # row back rather than decrypting a stale one.
            credential = (
                await get_credential_by_account(
                    GoogleDriveIntegration.provider, credential.provider_account_id
                )
                or credential
            )

    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise RuntimeError(
            f"Google Drive credential for account {credential.provider_account_id} "
            f"holds no access token"
        )
    return token
