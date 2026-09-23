"""Google Drive API calls used by the Drive integration.

OAuth operations live in :mod:`cognee.modules.integrations.google.client` so
Gmail and Drive cannot drift apart. This module keeps the Drive file API and
the compatibility exports used by existing Drive integrations and tests.
"""

import logging
from typing import Any

import aiohttp

from cognee.modules.integrations.google.client import (
    GoogleAuthError,
    _authorized_get,
    exchange_code,
    fetch_userinfo,
    refresh_access_token,
    revoke_token,
)

logger = logging.getLogger(__name__)

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVES_URL = "https://www.googleapis.com/drive/v3/drives"
MAX_FILE_BYTES = 1_000_000
_CHUNK_BYTES = 64 * 1024


async def list_files(
    access_token: str,
    page_token: str | None = None,
    *,
    query: str | None = "trashed = false",
    drive_id: str | None = None,
) -> dict[str, Any]:
    """Return one page of Drive files matching ``query``."""
    params = {
        "fields": (
            "nextPageToken, files(id, name, mimeType, webViewLink, modifiedTime, "
            "size, parents, driveId)"
        ),
        "orderBy": "modifiedTime desc",
        "pageSize": "100",
        "includeItemsFromAllDrives": "true",
        "supportsAllDrives": "true",
    }
    if query:
        params["q"] = query
    if drive_id:
        params.update({"corpora": "drive", "driveId": drive_id})
    if page_token:
        params["pageToken"] = page_token
    return await _authorized_get("file listing", DRIVE_FILES_URL, access_token, params)


async def list_folders(access_token: str, page_token: str | None = None) -> dict[str, Any]:
    """Return one page of folders and shared-drive roots."""
    return await list_files(
        access_token,
        page_token,
        query="mimeType = 'application/vnd.google-apps.folder' and trashed = false",
    )


async def list_drives(access_token: str, page_token: str | None = None) -> dict[str, Any]:
    """Return one page of shared drives visible to the connected account."""
    params = {
        "fields": "nextPageToken, drives(id,name)",
        "pageSize": "100",
    }
    if page_token:
        params["pageToken"] = page_token
    return await _authorized_get("shared-drive listing", DRIVES_URL, access_token, params)


async def export_file(access_token: str, file_id: str, mime_type: str) -> str:
    return await _authorized_text(
        "file export",
        f"{DRIVE_FILES_URL}/{file_id}/export",
        access_token,
        {"mimeType": mime_type},
    )


async def download_file(access_token: str, file_id: str) -> str:
    return await _authorized_text(
        "file download",
        f"{DRIVE_FILES_URL}/{file_id}",
        access_token,
        {"alt": "media", "supportsAllDrives": "true"},
    )


async def _authorized_text(
    operation: str, url: str, access_token: str, params: dict[str, str]
) -> str:
    chunks: list[bytes] = []
    received = 0
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session,
        session.get(
            url, params=params, headers={"Authorization": f"Bearer {access_token}"}
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"Google {operation} failed: HTTP {response.status}")
        async for chunk in response.content.iter_chunked(_CHUNK_BYTES):
            remaining = MAX_FILE_BYTES - received
            if remaining <= 0:
                logger.info(
                    "Google %s stopped at the %d byte ceiling; file is indexed truncated",
                    operation,
                    MAX_FILE_BYTES,
                )
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                logger.info(
                    "Google %s stopped at the %d byte ceiling; file is indexed truncated",
                    operation,
                    MAX_FILE_BYTES,
                )
                break
            chunks.append(chunk)
            received += len(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


__all__ = [
    "GoogleAuthError",
    "aiohttp",
    "download_file",
    "exchange_code",
    "export_file",
    "fetch_userinfo",
    "list_drives",
    "list_files",
    "list_folders",
    "refresh_access_token",
    "revoke_token",
]
