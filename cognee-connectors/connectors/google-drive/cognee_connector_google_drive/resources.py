"""What a Drive connection can enumerate, given an access token.

Split from cognee.modules.integrations.google_drive.adapter (cognee dev, 2026-10-26):
:func:`list_resources` is the body of ``GoogleDriveIntegration.list_resources`` there,
taken out of the host adapter so the contract object (``connector.py``) can offer the same
picker without a credential row. The entries keep the shape cognee's generic picker reads
(``id``, ``name``, ``description``, ``attributes``); the contract object wraps them in
``ConnectorResource``. Pure: nothing here imports cognee.
"""

from typing import Any

from cognee_connector_google_drive import client


async def shared_drive_ids(access_token: str) -> list[str]:
    """Every shared drive the account can see, across all pages, in listing order."""
    drive_ids: list[str] = []
    page_token = None
    while True:
        page = await client.list_drives(access_token, page_token)
        for drive in page.get("drives", []) or []:
            drive_id = str(drive.get("id") or "")
            if drive_id and drive_id not in drive_ids:
                drive_ids.append(drive_id)
        page_token = page.get("nextPageToken")
        if not page_token:
            return drive_ids


async def list_resources(access_token: str) -> list[dict[str, Any]]:
    """My Drive, the shared drives and every folder the account can see, for a picker."""
    token = access_token
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
