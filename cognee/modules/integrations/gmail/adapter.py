"""Google Gmail OAuth adapter."""

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
from cognee.modules.integrations.gmail import client as gmail_client
from cognee.modules.integrations.gmail.gmail_settings import GoogleGmailSettings, require
from cognee.modules.integrations.google import client
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_SCOPES = "openid email https://www.googleapis.com/auth/gmail.readonly"
_REFRESH_MARGIN = timedelta(minutes=5)


class GoogleGmailIntegration(OAuthIntegration):
    provider = "gmail"
    settings_cls = GoogleGmailSettings
    resource_selection_key = "selected_label_ids"

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": require("client_id"),
            "redirect_uri": require("redirect_uri"),
            "response_type": "code",
            "state": state,
            "scope": _SCOPES,
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
            logger.warning("Gmail account %s connected without a refresh token", subject)

        hosted_domain = userinfo.get("hd")
        return OAuthInstallation(
            provider_account_id=str(subject),
            token_payload=token_payload,
            provider_metadata={
                "email": userinfo.get("email"),
                "hosted_domain": hosted_domain,
                "account_type": "workspace" if hosted_domain else "personal",
                "scope": token_response.get("scope"),
                # Gmail is opt-in by default: a mailbox is more sensitive than
                # a folder list, so the user must choose labels before sync.
                "selected_label_ids": [],
            },
            account_label=userinfo.get("email"),
            scopes=token_response.get("scope"),
            token_expires_at=_expires_at(token_response.get("expires_in")),
        )

    def state_signing_secret(self) -> str:
        return require("state_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    async def on_installed(self, credential: IntegrationCredential) -> None:
        from cognee.modules.integrations.gmail.sync import sync_gmail

        await sync_gmail(credential)

    async def sync_now(self, credential: IntegrationCredential) -> None:
        from cognee.modules.integrations.gmail.sync import sync_gmail

        await sync_gmail(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        token = await access_token_for(credential)
        labels = await gmail_client.list_labels(token)
        return [
            {
                "id": str(label["id"]),
                "name": str(label.get("name") or label["id"]),
                "description": None,
                "attributes": {
                    "type": label.get("type"),
                    "messages_total": label.get("messagesTotal"),
                },
            }
            for label in labels.get("labels", []) or []
            if label.get("id")
        ]

    def dataset_name(self, credential: IntegrationCredential) -> str:
        from cognee.modules.integrations.gmail.sync import dataset_name_for_account

        metadata = credential.provider_metadata or {}
        return dataset_name_for_account(
            str(metadata.get("email") or ""), str(credential.provider_account_id)
        )

    async def refresh(self, credential: IntegrationCredential) -> None:
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"Gmail credential for account {credential.provider_account_id} "
                "holds no refresh token; the account must reconnect"
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
            raise

        await upsert_credential(
            provider=self.provider,
            user_id=credential.user_id,
            provider_account_id=credential.provider_account_id,
            token_payload={
                "access_token": refreshed["access_token"],
                "refresh_token": refresh_token,
            },
            workspace_id=credential.workspace_id,
            account_label=credential.account_label,
            auth_type=credential.auth_type,
            scopes=refreshed.get("scope") or credential.scopes,
            provider_metadata=credential.provider_metadata,
            token_expires_at=_expires_at(refreshed.get("expires_in")),
        )

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        try:
            token_payload = decrypt_token_payload(credential)
            token = token_payload.get("refresh_token") or token_payload.get("access_token")
            if token:
                await client.revoke_token(token)
        except Exception:  # disconnect must not be blocked by Google
            logger.exception(
                "Gmail token revoke failed for account %s", credential.provider_account_id
            )


def _expires_at(expires_in: Any) -> datetime | None:
    if not expires_in:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))


async def access_token_for(credential: IntegrationCredential) -> str:
    expires_at = credential.token_expires_at
    if expires_at is not None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at - _REFRESH_MARGIN <= datetime.now(timezone.utc):
            await GoogleGmailIntegration().refresh(credential)
            credential = (
                await get_credential_by_account(
                    GoogleGmailIntegration.provider, credential.provider_account_id
                )
                or credential
            )
    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise RuntimeError(
            f"Gmail credential for account {credential.provider_account_id} holds no access token"
        )
    return token
