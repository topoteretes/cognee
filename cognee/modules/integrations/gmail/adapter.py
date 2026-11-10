"""Google Gmail OAuth adapter."""

from typing import Any

from cognee.modules.integrations.gmail import client as gmail_client
from cognee.modules.integrations.gmail.gmail_settings import (
    GoogleGmailSettings,
    google_gmail_settings,
)
from cognee.modules.integrations.gmail.sync import GMAIL_DATASET_PREFIX
from cognee.modules.integrations.google.oauth_integration import GoogleOAuthIntegration
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


class GoogleGmailIntegration(GoogleOAuthIntegration):
    provider = "gmail"
    settings_cls = GoogleGmailSettings
    settings = google_gmail_settings
    resource_selection_key = "selected_label_ids"
    display_name = "Gmail"
    scope = "openid email https://www.googleapis.com/auth/gmail.readonly"
    dataset_prefix = GMAIL_DATASET_PREFIX

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


async def access_token_for(credential: IntegrationCredential) -> str:
    return await GoogleGmailIntegration().access_token(credential)
