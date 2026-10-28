"""Google Drive OAuth adapter."""

from typing import Any

from cognee.modules.integrations.google.oauth_integration import GoogleOAuthIntegration
from cognee.modules.integrations.google_drive import client
from cognee.modules.integrations.google_drive.google_drive_settings import (
    GoogleDriveSettings,
    google_drive_settings,
)
from cognee.modules.integrations.google_drive.sync import GOOGLE_DRIVE_DATASET_PREFIX
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


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
        from cognee.modules.integrations.google_drive.sync import sync_drive

        await sync_drive(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        token = await access_token_for(credential)
        resources = [
            {
                "id": "root",
                "name": "My Drive",
                "description": "Files in the account root, including nested folders",
                "attributes": {"mime_type": "application/vnd.google-apps.folder", "root": True},
            }
        ]
        resource_ids = {"root"}
        drive_page_token = None
        while True:
            page = await client.list_drives(token, drive_page_token)
            for drive in page.get("drives", []) or []:
                drive_id = str(drive.get("id") or "")
                if not drive_id or drive_id in resource_ids:
                    continue
                resource_ids.add(drive_id)
                resources.append(
                    {
                        "id": drive_id,
                        "name": str(drive.get("name") or "Shared drive"),
                        "description": "Files in this shared drive",
                        "attributes": {
                            "mime_type": "application/vnd.google-apps.folder",
                            "shared_drive": True,
                            "root": True,
                        },
                    }
                )
            drive_page_token = page.get("nextPageToken")
            if not drive_page_token:
                break

        page_token = None
        while True:
            page = await client.list_folders(token, page_token)
            for folder in page.get("files", []) or []:
                folder_id = str(folder.get("id") or "")
                if not folder_id or folder_id in resource_ids:
                    continue
                resource_ids.add(folder_id)
                resources.append(
                    {
                        "id": folder_id,
                        "name": str(folder.get("name") or "Untitled folder"),
                        "description": None,
                        "attributes": {
                            "mime_type": folder.get("mimeType"),
                            "drive_id": folder.get("driveId"),
                        },
                    }
                )
            page_token = page.get("nextPageToken")
            if not page_token:
                return resources


async def access_token_for(credential: IntegrationCredential) -> str:
    return await GoogleDriveIntegration().access_token(credential)
