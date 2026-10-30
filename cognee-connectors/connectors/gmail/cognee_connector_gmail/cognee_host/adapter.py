"""Moved from cognee.modules.integrations.gmail.adapter (cognee dev, 2026-10-26).

Google Gmail OAuth adapter.
"""

from typing import Any

from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

from cognee_connector_gmail import resources
from cognee_connector_gmail.cognee_host.sync import GMAIL_DATASET_PREFIX
from cognee_connector_gmail.settings import GoogleGmailSettings, google_gmail_settings
from cognee_connector_google.cognee_host.oauth_integration import GoogleOAuthIntegration


class GoogleGmailIntegration(GoogleOAuthIntegration):
    provider = "gmail"
    settings_cls = GoogleGmailSettings
    settings = google_gmail_settings
    resource_selection_key = "selected_label_ids"
    display_name = "Gmail"
    scope = "openid email https://www.googleapis.com/auth/gmail.readonly"
    dataset_prefix = GMAIL_DATASET_PREFIX

    async def sync_now(self, credential: IntegrationCredential) -> None:
        from cognee_connector_gmail.cognee_host.sync import sync_gmail

        await sync_gmail(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        return await resources.list_resources(await access_token_for(credential))


async def access_token_for(credential: IntegrationCredential) -> str:
    return await GoogleGmailIntegration().access_token(credential)
