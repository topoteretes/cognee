"""Notion as an ``OAuthIntegration`` adapter.

One authorization covers exactly one workspace, so a second workspace is a second
connection, keyed on the workspace id. Notion's token endpoints authenticate the app
with HTTP Basic, and the token response names the workspace that authorized, so no
extra call is needed to learn the account.

Tokens: Notion's docs describe a rotating refresh token and show no ``expires_in``.
A response that carries no expiry is stored without one and used as is; a response
that carries one is refreshed five minutes ahead. Every refresh returns a new refresh
token and the one just spent stops working, so the stored one is replaced each time.
Notion has a revoke endpoint, so a disconnect revokes both tokens.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    require_active_credential,
    revoke_credential_if_current,
    update_refreshed_credential,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.notion import client
from cognee.modules.integrations.notion.notion_settings import NotionSettings, require
from cognee.modules.integrations.notion.sync import (
    SELECTION_KEY,
    dataset_name_for_credential,
    resource_id,
    sync_notion,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("notion_integration")

_AUTHORIZE_URL = f"{client.API_BASE}/oauth/authorize"

# Refresh this far before the token actually dies, so a sync that takes a while does
# not expire halfway through its own walk.
_REFRESH_MARGIN = timedelta(minutes=5)

# One lock per workspace, so the picker and a background sync that both find the token
# expiring do not each spend the same refresh token. Notion rotates it, so the second
# call would fail and could revoke a healthy connection. Process-local, like the sync
# guard in ``ingestion``; the compare-and-swap in ``update_refreshed_credential`` covers
# a second process. One entry per connected workspace, never expired.
_refresh_locks: dict[str, asyncio.Lock] = {}


class NotionIntegration(OAuthIntegration):
    provider = "notion"
    settings_cls = NotionSettings
    resource_selection_key = SELECTION_KEY
    # Nothing syncs until the user picks roots: ingesting a whole workspace costs LLM tokens.
    resource_selection_allows_all = False

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": require("client_id"),
            "redirect_uri": require("redirect_uri"),
            "response_type": "code",
            "owner": "user",
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

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        workspace_id = token_response.get("workspace_id")
        if not workspace_id:
            raise ValueError("Notion token response carries no workspace_id")

        token_payload: dict[str, Any] = {"access_token": token_response["access_token"]}
        if token_response.get("refresh_token"):
            token_payload["refresh_token"] = token_response["refresh_token"]
        return OAuthInstallation(
            provider_account_id=str(workspace_id),
            token_payload=token_payload,
            provider_metadata={
                "bot_id": token_response.get("bot_id"),
                "workspace_icon": token_response.get("workspace_icon"),
                # A new connection must opt into a selection before importing data.
                # complete_installation preserves existing choices on reconnect.
                SELECTION_KEY: [],
            },
            account_label=token_response.get("workspace_name"),
            token_expires_at=_expires_at(token_response.get("expires_in")),
        )

    def state_signing_secret(self) -> str:
        return require("state_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Connecting updates authorization only; sync starts via the Refresh action."""
        return

    def dataset_name(self, credential: IntegrationCredential) -> str:
        return dataset_name_for_credential(credential)

    async def sync_now(self, credential: IntegrationCredential) -> None:
        await sync_notion(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        """The roots a user can pick: shared pages and data sources whose parent is not shared.

        A page or data source inside another shared one is read by that one's walk, so
        offering it too would have the sync read it twice. Search only knows the parents
        it returns: a page inside a block (a toggle or a column) of a shared page is
        offered here, and the sync drops it when it walks the parent chain.
        """
        hits = await client.search_shared(await access_token_for(credential))
        shared = {client.canonical_id(hit["id"]) for hit in hits}
        return [
            {
                "id": resource_id(hit["kind"], hit["id"]),
                "name": hit["title"] or "Untitled",
                "description": None,
                "attributes": {
                    "kind": hit["kind"],
                    "parent_kind": hit["parent_kind"],
                    "parent_id": hit["parent_id"],
                },
            }
            for hit in hits
            if not hit["parent_id"] or client.canonical_id(hit["parent_id"]) not in shared
        ]

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Rotate the pair in place.

        An ``invalid_grant`` means the workspace removed cognee's access at Notion.
        Nothing tells us that otherwise, since this connector receives no webhooks, so
        the local credential is revoked here: the alternative is a connection that shows
        as healthy forever while every sync fails.
        """
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"Notion credential for workspace {credential.provider_account_id} "
                "holds no refresh token; the workspace must reconnect"
            )
        try:
            refreshed = await client.refresh_access_token(
                refresh_token,
                client_id=require("client_id"),
                client_secret=require("client_secret"),
            )
        except client.NotionError as error:
            if error.code == "invalid_grant":
                await revoke_credential_if_current(credential)
                logger.warning(
                    "Notion access for workspace %s was revoked at the provider; "
                    "local credential revoked",
                    credential.provider_account_id,
                )
            raise

        await update_refreshed_credential(
            credential,
            # Notion rotates: the response carries the pair to keep. If it carries no new
            # refresh token the old one may still work, and storing nothing would lose it.
            token_payload={
                "access_token": refreshed["access_token"],
                "refresh_token": refreshed.get("refresh_token") or refresh_token,
            },
            scopes=credential.scopes,
            token_expires_at=_expires_at(refreshed.get("expires_in")),
        )

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort revoke of both tokens.

        The local revoke is the actual access cut-off, and a network blip must never
        block a disconnect, so one token's failure does not stop the other's revoke.
        """
        token_payload = decrypt_token_payload(credential)
        for name in ("access_token", "refresh_token"):
            token = token_payload.get(name)
            if not token:
                continue
            try:
                await client.revoke_token(
                    token, client_id=require("client_id"), client_secret=require("client_secret")
                )
            except Exception:  # disconnect must not be blocked by Notion
                logger.exception(
                    "Notion %s revoke failed for workspace %s",
                    name,
                    credential.provider_account_id,
                )


async def access_token_for(credential: IntegrationCredential) -> str:
    """A bearer token, refreshed first when it is within five minutes of its expiry."""
    credential = await require_active_credential(credential)
    if _is_expiring(credential):
        async with _refresh_locks.setdefault(str(credential.provider_account_id), asyncio.Lock()):
            # A caller that held the lock first may have rotated the pair already, so
            # read the row back before spending the refresh token.
            credential = await require_active_credential(credential)
            if _is_expiring(credential):
                await NotionIntegration().refresh(credential)
                # refresh() writes through its own session, so the instance handed in
                # still carries the pre-rotation ciphertext. Read the row back.
                credential = await require_active_credential(credential)

    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise RuntimeError(
            f"Notion credential for workspace {credential.provider_account_id} "
            "holds no access token"
        )
    return token


def _is_expiring(credential: IntegrationCredential) -> bool:
    expires_at = credential.token_expires_at
    if expires_at is None:
        return False
    # A naive timestamp comes back from SQLite, which stores no timezone; it is
    # written as UTC, so that is what it is read as.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at - _REFRESH_MARGIN <= datetime.now(timezone.utc)


def _expires_at(expires_in: Any) -> datetime | None:
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in) if expires_in else None
