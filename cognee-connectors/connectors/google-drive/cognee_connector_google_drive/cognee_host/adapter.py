"""Moved from cognee.modules.integrations.google_drive.adapter (cognee dev, 2026-10-26).

Google Drive OAuth adapter.
"""

from typing import Any

from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

from cognee_connector_google.cognee_host.oauth_integration import GoogleOAuthIntegration
from cognee_connector_google_drive import resources
from cognee_connector_google_drive.cognee_host.sync import GOOGLE_DRIVE_DATASET_PREFIX
from cognee_connector_google_drive.settings import GoogleDriveSettings, google_drive_settings


class GoogleDriveIntegration(GoogleOAuthIntegration):
    provider = "google_drive"
    settings_cls = GoogleDriveSettings
    settings = google_drive_settings
    resource_selection_key = "selected_folder_ids"
    display_name = "Google Drive"
    # drive.readonly is a restricted scope and needs Google's OAuth verification
    # before a public launch; openid and email are what identify the account the
    # credential is keyed on.
    scope = "openid email https://www.googleapis.com/auth/drive.readonly"
    dataset_prefix = GOOGLE_DRIVE_DATASET_PREFIX

    async def sync_now(self, credential: IntegrationCredential) -> None:
        from cognee_connector_google_drive.cognee_host.sync import sync_drive

        await sync_drive(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        return await resources.list_resources(await access_token_for(credential))


async def access_token_for(credential: IntegrationCredential) -> str:
    return await GoogleDriveIntegration().access_token(credential)
