"""Log-safe Dropbox OAuth, account and folder calls for the Dropbox integration.

Copied from cognee.modules.integrations.dropbox.client (cognee dev, 2026-11-12). Sessions are
plain ``aiohttp.ClientSession`` objects: cognee's ``client_session`` wrapper also refuses the
call when the server sets ``ALLOW_HTTP_REQUESTS`` to a disabling value. That is a host policy;
cognee's integrations router and sync runner check it themselves, any other host applies its
own.

Every failure is a :class:`DropboxIntegrationError` carrying the operation and Dropbox's error
tag or the HTTP status, never a message, a body or a token: these errors end up in logs.
"""

import asyncio
import re
from typing import Any

import aiohttp

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


async def _post(
    operation: str, url: str, *, session: Any = None, void: bool = False, **kwargs: Any
) -> dict[str, Any]:
    """One POST, its JSON object on a 200, else the typed error. Redirects are refused.

    ``session`` is an open ``client_session`` to send through (a paged listing reuses one);
    without it the call opens and closes its own. ``void``: a route that answers no result
    (Dropbox sends ``null`` or nothing); its 200 is success, returned as ``{}``.
    """
    try:
        if session is None:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as own:
                status, body = await _send(own, url, kwargs)
        else:
            status, body = await _send(session, url, kwargs)
    except asyncio.TimeoutError:
        raise DropboxIntegrationError(operation, "request_timeout") from None
    except (aiohttp.ClientError, OSError):
        raise DropboxIntegrationError(operation, "network_error") from None
    if status == 200 and void:
        return {}
    if status != 200 or not isinstance(body, dict):
        raise DropboxIntegrationError(operation, _code(body, status))
    return body


async def _send(session: Any, url: str, kwargs: dict[str, Any]) -> tuple[int, Any]:
    async with session.post(url, allow_redirects=False, **kwargs) as response:
        try:
            return response.status, await response.json(content_type=None)
        except (aiohttp.ClientError, ValueError):
            return response.status, None


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
    await _post("token revoke", REVOKE_URL, void=True, data=b"null", headers=_bearer(access_token))


async def get_current_account(access_token: str) -> dict[str, Any]:
    return await _post("account lookup", ACCOUNT_URL, data=b"null", headers=_bearer(access_token))


async def get_folder(
    access_token: str, folder_id: str, *, session: Any = None
) -> dict[str, Any] | None:
    """A folder's metadata, or None when it is gone or not a folder. ``session``: see _post."""
    try:
        meta = await _post(
            "folder lookup",
            METADATA_URL,
            session=session,
            json={"path": folder_id},
            headers=_bearer(access_token),
        )
    except DropboxIntegrationError as error:
        if error.code == "path":
            return None
        raise
    return meta if meta.get(".tag") == "folder" else None


async def get_folders(access_token: str, folder_ids: list[str]) -> list[dict[str, Any] | None]:
    """``get_folder`` for each id, in order, through one HTTP session."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        return [await get_folder(access_token, f, session=session) for f in folder_ids]


async def list_folders(access_token: str, parent: str | None = None) -> list[dict[str, Any]]:
    """The folders directly inside ``parent`` (a folder id; None: the top of the account), for
    the picker, with their id and path.

    One level per call, not the whole account: a picker opens a folder to see its children.
    Cost: the files of that one level are listed too (Dropbox has no folders-only listing),
    one ``list_folder`` page of up to 2000 entries per request, through one HTTP session.
    """
    headers = _bearer(access_token)
    folders: list[dict[str, Any]] = []
    seen: set[str] = set()
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        page = await _post(
            "folder listing",
            LIST_URL,
            session=session,
            json={"path": parent or "", "recursive": False, "limit": LIST_LIMIT},
            headers=headers,
        )
        while True:
            folders += [e for e in page.get("entries") or [] if e.get(".tag") == "folder"]
            if not page.get("has_more"):
                return folders
            cursor = page.get("cursor")
            if not isinstance(cursor, str) or cursor in seen:
                raise DropboxIntegrationError("folder listing", "cursor_repeated")
            seen.add(cursor)
            page = await _post(
                "folder listing",
                CONTINUE_URL,
                session=session,
                json={"cursor": cursor},
                headers=headers,
            )
