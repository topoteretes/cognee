"""Log-safe Dropbox OAuth, account and folder calls for the Dropbox integration.

Every failure is a :class:`DropboxIntegrationError` carrying the operation and Dropbox's error
tag or the HTTP status, never a message, a body or a token: these errors end up in logs.
"""

import asyncio
import re
from typing import Any

import aiohttp

from cognee.modules.integrations.outbound import client_session

API = "https://api.dropboxapi.com"
TOKEN_URL = f"{API}/oauth2/token"
REVOKE_URL = f"{API}/2/auth/token/revoke"
ACCOUNT_URL = f"{API}/2/users/get_current_account"
METADATA_URL = f"{API}/2/files/get_metadata"
LIST_URL = f"{API}/2/files/list_folder"
CONTINUE_URL = f"{API}/2/files/list_folder/continue"
TIMEOUT = aiohttp.ClientTimeout(total=30)
# list_folder's documented (approximate) maximum page size.
LIST_LIMIT = 2000
_CODE = re.compile(r"[a-z_]{1,64}")


class DropboxIntegrationError(RuntimeError):
    """A Dropbox failure that never includes response content."""

    def __init__(self, operation: str, code: str):
        super().__init__(f"Dropbox {operation} failed: {code}")
        self.operation = operation
        self.code = code


def _code(body: Any, status: int) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        tag = error.get(".tag") if isinstance(error, dict) else error
        if isinstance(tag, str) and _CODE.fullmatch(tag):
            return tag
    return f"http_{status}"


async def _post(operation: str, url: str, **kwargs: Any) -> dict[str, Any]:
    """One POST, its JSON object on a 200, else the typed error. Redirects are refused."""
    try:
        async with (
            client_session("dropbox", timeout=TIMEOUT) as session,
            session.post(url, allow_redirects=False, **kwargs) as response,
        ):
            status = response.status
            try:
                body = await response.json(content_type=None)
            except (aiohttp.ClientError, ValueError):
                body = None
    except asyncio.TimeoutError:
        raise DropboxIntegrationError(operation, "request_timeout") from None
    except (aiohttp.ClientError, OSError):
        raise DropboxIntegrationError(operation, "network_error") from None
    if status != 200 or not isinstance(body, dict):
        raise DropboxIntegrationError(operation, _code(body, status))
    return body


def _bearer(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}


async def exchange_code(
    code: str, *, client_id: str, client_secret: str, redirect_uri: str
) -> dict[str, Any]:
    """Exchange an authorization code. ``account_id`` names the account the token is for."""
    body = await _post(
        "code exchange",
        TOKEN_URL,
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
        auth=aiohttp.BasicAuth(client_id, client_secret),
    )
    if not isinstance(body.get("access_token"), str) or not isinstance(body.get("account_id"), str):
        raise DropboxIntegrationError("code exchange", "incomplete_token_response")
    return body


async def refresh_access_token(
    refresh_token: str, *, client_id: str, client_secret: str
) -> dict[str, Any]:
    """A new short-lived access token. Dropbox does not rotate the refresh token."""
    body = await _post(
        "token refresh",
        TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        auth=aiohttp.BasicAuth(client_id, client_secret),
    )
    if not isinstance(body.get("access_token"), str):
        raise DropboxIntegrationError("token refresh", "no_access_token")
    return body


async def revoke_token(access_token: str) -> None:
    """Revoke the access token, which also revokes its refresh token (Dropbox OAuth guide)."""
    await _post("token revoke", REVOKE_URL, data=b"null", headers=_bearer(access_token))


async def get_current_account(access_token: str) -> dict[str, Any]:
    return await _post("account lookup", ACCOUNT_URL, data=b"null", headers=_bearer(access_token))


async def get_folder(access_token: str, folder_id: str) -> dict[str, Any] | None:
    """A folder's metadata, or None when it is gone or not a folder."""
    try:
        meta = await _post(
            "folder lookup", METADATA_URL, json={"path": folder_id}, headers=_bearer(access_token)
        )
    except DropboxIntegrationError as error:
        if error.code == "path":
            return None
        raise
    return meta if meta.get(".tag") == "folder" else None


async def list_folders(access_token: str) -> list[dict[str, Any]]:
    """Every folder the account can see, for the picker, with its id and path."""
    page = await _post(
        "folder listing",
        LIST_URL,
        json={"path": "", "recursive": True, "limit": LIST_LIMIT},
        headers=_bearer(access_token),
    )
    folders: list[dict[str, Any]] = []
    seen: set[str] = set()
    while True:
        folders += [e for e in page.get("entries") or [] if e.get(".tag") == "folder"]
        if not page.get("has_more"):
            return folders
        cursor = page.get("cursor")
        if not isinstance(cursor, str) or cursor in seen:
            raise DropboxIntegrationError("folder listing", "cursor_repeated")
        seen.add(cursor)
        page = await _post(
            "folder listing", CONTINUE_URL, json={"cursor": cursor}, headers=_bearer(access_token)
        )
