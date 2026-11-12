"""Dropbox as an ``OAuthIntegration`` adapter.

One authorization is one Dropbox account, keyed on its ``account_id`` (``dbid:...``), which
Dropbox calls unique and stable; the email is only a label. ``token_access_type=offline``
makes Dropbox issue a refresh token next to the four-hour access token. The refresh token
does not expire or rotate, and revoking the access token revokes both. The app secret
authenticates the code exchange; PKCE is not used, because the base flow has no place to
keep a code verifier between ``authorize_url`` and the callback.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from cognee.modules.integrations import credentials as credential_store
from cognee.modules.integrations.base import (
    OAuthInstallation,
    OAuthIntegration,
    ResourceParentError,
    ResourceParentNotFoundError,
)
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    require_active_credential,
    revoke_credential_if_current,
    update_refreshed_credential,
)
from cognee.modules.integrations.dropbox import client
from cognee.modules.integrations.dropbox.dropbox_settings import DropboxSettings, require
from cognee.modules.integrations.dropbox.sync import (
    SELECTION_KEY,
    check_selection_shape,
    dataset_name_for_credential,
    refuse_nested_selection,
    sync_dropbox,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.connectors.dropbox import valid_folder_id

logger = get_logger("dropbox_integration")

_AUTHORIZE_URL = "https://www.dropbox.com/oauth2/authorize"
SCOPES = "files.metadata.read files.content.read account_info.read"
# One lock per account, so the picker and a sync that both find the token expiring refresh
# once. Process-local, like the sync guard in ``ingestion``.
_refresh_locks: dict[str, asyncio.Lock] = {}


class DropboxIntegration(OAuthIntegration):
    provider = "dropbox"
    settings_cls = DropboxSettings
    resource_selection_key = SELECTION_KEY
    # Nothing syncs until the user picks folders: a whole Dropbox costs LLM tokens.
    resource_selection_allows_all = False
    # The picker opens one folder at a time: an account can hold too many to list at once.

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": require("client_id"),
            "redirect_uri": require("redirect_uri"),
            "response_type": "code",
            "token_access_type": "offline",
            "scope": SCOPES,
            "state": state,
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
        """Exchange the code, then read the account's email for the label and dataset name."""
        token_response = await self.exchange_code(code)
        account = await client.get_current_account(token_response["access_token"])
        if account.get("account_id") != token_response["account_id"]:
            raise ValueError("Dropbox token and account lookup name different accounts")
        return {**token_response, "email": account.get("email")}

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        account_id = token_response.get("account_id")
        if not isinstance(account_id, str) or not account_id.startswith("dbid:"):
            raise ValueError("Dropbox token response carries no account_id")
        token_payload: dict[str, Any] = {"access_token": token_response["access_token"]}
        if token_response.get("refresh_token"):
            token_payload["refresh_token"] = token_response["refresh_token"]
        return OAuthInstallation(
            provider_account_id=account_id,
            token_payload=token_payload,
            provider_metadata={
                "email": token_response.get("email"),
                # A new connection must opt into a selection before importing data;
                # complete_installation keeps an existing choice on reconnect.
                SELECTION_KEY: [],
            },
            account_label=token_response.get("email"),
            scopes=token_response.get("scope"),
            token_expires_at=_expires_at(token_response.get("expires_in")),
        )

    def state_signing_secret(self) -> str:
        return require("state_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Connecting updates authorization only; a sync starts with Sync now or a schedule."""
        return

    def dataset_name(self, credential: IntegrationCredential) -> str:
        return dataset_name_for_credential(credential)

    async def sync_now(self, credential: IntegrationCredential) -> None:
        await sync_dropbox(credential)

    def invalid_resource_ids(self, resource_ids: list[str]) -> list[str]:
        """Ids that are not Dropbox folder ids (``id:...``); the source refuses the same ones."""
        from cognee.tasks.ingestion.connectors.dropbox import valid_folder_id

        return [resource_id for resource_id in resource_ids if not valid_folder_id(resource_id)]

    async def validate_resource_selection(
        self, credential: IntegrationCredential, resource_ids: list[str]
    ) -> None:
        """Every folder id resolves at Dropbox, and no folder sits inside another (its files
        would be two documents). A single folder is checked too: an id Dropbox cannot resolve
        would otherwise be stored and count as gone on every sync."""
        folder_ids = check_selection_shape(resource_ids)
        if folder_ids:
            await refuse_nested_selection(
                credential, await access_token_for(credential), folder_ids, refuse_missing=True
            )

    async def list_resources(
        self, credential: IntegrationCredential, *, parent: str | None = None
    ) -> list[dict[str, Any]]:
        """The folders one level under ``parent`` (a folder id; None: the top of the account),
        by id, with their path for the picker. A parent that is not a folder id is a 422
        (``ResourceParentError``); one Dropbox does not find, or a file, a 404."""
        if parent is not None and not valid_folder_id(parent):
            # Checked before the token is used: a path, an ns: id or a control character
            # never reaches Dropbox.
            raise ResourceParentError("parent must be a Dropbox folder id like id:...")
        try:
            folders = await client.list_folders(await access_token_for(credential), parent=parent)
        except client.DropboxIntegrationError as error:
            if parent is None or error.code != "path":
                raise
            # list_folder's path error: the parent is gone, not visible, or not a folder.
            raise ResourceParentNotFoundError("That Dropbox folder could not be found") from None
        return [
            {
                "id": folder["id"],
                "name": folder.get("path_display") or folder.get("name") or "Untitled folder",
                "description": None,
                "attributes": {"path": folder.get("path_display")},
            }
            for folder in sorted(folders, key=lambda f: str(f.get("path_lower")))
            if isinstance(folder.get("id"), str)
        ]

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Renew the access token. ``invalid_grant`` means the account removed the app."""
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"Dropbox credential for account {credential.provider_account_id} "
                "holds no refresh token; the account must reconnect"
            )
        try:
            refreshed = await client.refresh_access_token(
                refresh_token,
                client_id=require("client_id"),
                client_secret=require("client_secret"),
            )
        except client.DropboxIntegrationError as error:
            if error.code == "invalid_grant":
                await revoke_credential_if_current(credential)
                logger.warning(
                    "Dropbox access for account %s was revoked at the provider; "
                    "local credential revoked",
                    credential.provider_account_id,
                )
            raise
        expires_at = _expires_at(refreshed.get("expires_in"))
        if expires_at is None:
            # Stored without an expiry the token would be cached and never renewed.
            raise client.DropboxIntegrationError("token refresh", "no_expires_in")
        await update_refreshed_credential(
            credential,
            # Dropbox does not rotate the refresh token, so the stored one is kept.
            token_payload={
                "access_token": refreshed["access_token"],
                "refresh_token": refresh_token,
            },
            scopes=refreshed.get("scope") or credential.scopes,
            token_expires_at=expires_at,
        )

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort revoke; the local revoke is the actual cut-off.

        The router revokes the credential locally first, so ``access_token_for`` (which checks
        it is active and writes a refreshed token) cannot be used. An access token past or near
        its expiry cannot revoke itself, so a fresh one is taken from the refresh token, in
        memory only, and revoking it revokes the refresh token with it.
        """
        token_payload = decrypt_token_payload(credential)
        token = token_payload.get("access_token")
        refresh_token = token_payload.get("refresh_token")
        try:
            if refresh_token and (
                not token or credential_store.token_is_expiring(credential.token_expires_at, _now())
            ):
                refreshed = await client.refresh_access_token(
                    refresh_token,
                    client_id=require("client_id"),
                    client_secret=require("client_secret"),
                )
                token = refreshed["access_token"]
            if token:
                await client.revoke_token(token)
        except client.DropboxIntegrationError as error:
            # The app stays listed in the user's Dropbox settings until they remove it.
            logger.warning(
                "Dropbox token revoke failed for account %s: %s",
                credential.provider_account_id,
                error.code,
            )


async def access_token_for(credential: IntegrationCredential) -> str:
    """A bearer token, refreshed first when it is within five minutes of its expiry."""
    token, _expires_at_value, _ciphertext = await access_token_and_expiry(credential)
    return token


async def access_token_and_expiry(
    credential: IntegrationCredential, rejected_ciphertext: bytes | None = None
) -> tuple[str, datetime | None, bytes]:
    """The bearer token, its expiry and the stored ciphertext it came from.

    Renewed first when it is near its expiry, or when ``rejected_ciphertext`` (the stored
    token Dropbox just answered 401 for) is still what the row holds.
    """
    credential = await credential_store.refreshed_credential(
        credential,
        DropboxIntegration().refresh,
        locks=_refresh_locks,
        read=require_active_credential,
        now=_now,
        rejected_ciphertext=rejected_ciphertext,
    )
    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise RuntimeError(
            f"Dropbox credential for account {credential.provider_account_id} holds no access token"
        )
    return token, credential.token_expires_at, credential.ciphertext


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at(expires_in: Any) -> datetime | None:
    if not isinstance(expires_in, int) or isinstance(expires_in, bool) or expires_in <= 0:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in)
