"""Log-safe Notion OAuth, search and parent lookups for the Notion integration.

Notion's token endpoints authenticate the app with HTTP Basic (client id and secret)
and take a JSON body. Every failure is a :class:`NotionError` carrying Notion's error
code or the HTTP status, never a message or a body: Notion's messages can quote
workspace content, and these errors end up in logs.
"""

import asyncio
import math
import re
from base64 import b64encode
from typing import Any

import aiohttp

from cognee.shared.logging_utils import get_logger

logger = get_logger("notion_integration")

API_BASE = "https://api.notion.com/v1"
TOKEN_URL = f"{API_BASE}/oauth/token"
REVOKE_URL = f"{API_BASE}/oauth/revoke"
SEARCH_URL = f"{API_BASE}/search"
USERS_ME_URL = f"{API_BASE}/users/me"
# Codes by which Notion says the token itself is no good.
UNAUTHORIZED_CODES = frozenset({"unauthorized", "http_401"})
# The version ``notion_source`` is pinned to. The shapes read below (data_source
# objects, ``in_trash``) are that version's.
NOTION_VERSION = "2025-09-03"
TIMEOUT = aiohttp.ClientTimeout(total=30)

SEARCH_PAGE_SIZE = 100
# The picker holds an HTTP request open for the whole search walk, and Notion answers
# about 3 requests a second. Search also returns every row of every shared database as
# a page, so a workspace sharing one 20,000-row database is 200 sequential calls. The
# walk is bounded by pages and by a deadline and ends in an error, never a prefix of
# the list: a truncated list would hide roots the user shared.
MAX_SEARCH_PAGES = 100
SEARCH_DEADLINE_SECONDS = 60
# The longest a token lifetime can plausibly be. Anything longer is not a lifetime, and
# one too large for a ``timedelta`` would fail after Notion has already rotated the pair.
_MAX_EXPIRES_IN = 366 * 24 * 60 * 60
# Notion answers a request over its rate (about 3 a second) with 429 and says how long
# to wait. Waiting is bounded, so a stuck API ends the call instead of holding the
# picker request open.
MAX_RATE_LIMIT_RETRIES = 3
MAX_RETRY_AFTER_SECONDS = 15
DEFAULT_RETRY_AFTER_SECONDS = 1
# What a provider error code looks like (``invalid_grant``, ``rate_limited``).
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
# The URL path of each kind of object a parent lookup can land on.
_OBJECT_PATH = {
    "page": "pages",
    "data_source": "data_sources",
    "database": "databases",
    "block": "blocks",
}
# Notion's parent types that name an object to fetch next, as the kind ``get_parent`` returns.
_PARENT_KINDS = {
    "page_id": "page",
    "data_source_id": "data_source",
    "database_id": "database",
    "block_id": "block",
}


class NotionError(RuntimeError):
    """A stable Notion failure that never includes response content."""

    def __init__(self, operation: str, code: str):
        self.code = code
        super().__init__(f"Notion {operation} failed: {code}")


def canonical_id(value: str) -> str:
    """The lowercase dashed-free form of a Notion id, so two spellings compare equal."""
    return value.replace("-", "").lower()


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {"Notion-Version": NOTION_VERSION, **(extra or {})}


def _bearer(access_token: str) -> dict[str, str]:
    return _headers({"Authorization": f"Bearer {access_token}"})


def _app_headers(client_id: str, client_secret: str) -> dict[str, str]:
    """Headers for the token endpoints, which authenticate the app, not a user."""
    credentials = b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return _headers({"Authorization": f"Basic {credentials}"})


def _retry_delay(value: str | None) -> float:
    """Seconds to wait after a 429: Notion's ``Retry-After``, bounded."""
    try:
        seconds = float(value) if value is not None else DEFAULT_RETRY_AFTER_SECONDS
    except ValueError:
        seconds = DEFAULT_RETRY_AFTER_SECONDS
    # ``float("nan")`` parses, and ``asyncio.sleep(nan)`` never wakes.
    if not math.isfinite(seconds):
        seconds = DEFAULT_RETRY_AFTER_SECONDS
    return min(max(seconds, 0), MAX_RETRY_AFTER_SECONDS)


def _code(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and _ERROR_CODE.fullmatch(value) else fallback


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


async def _request(
    operation: str,
    method: str,
    url: str,
    session: aiohttp.ClientSession,
    **kwargs: Any,
) -> tuple[int, Any]:
    """Send one request and return the status with the decoded JSON body.

    A 429 (a request Notion did not process, so sending it again is safe) is waited
    out up to :data:`MAX_RATE_LIMIT_RETRIES` times and then returned for the caller
    to report.
    """
    attempt = 0
    while True:
        try:
            async with session.request(method, url, **kwargs) as response:
                status = response.status
                if status == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                    delay = _retry_delay(response.headers.get("Retry-After"))
                else:
                    try:
                        return status, await response.json(content_type=None)
                    except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
                        raise NotionError(operation, f"http_{status}") from None
        except asyncio.TimeoutError:
            raise NotionError(operation, "request_timeout") from None
        except (aiohttp.ClientError, OSError):
            raise NotionError(operation, "network_error") from None
        attempt += 1
        await asyncio.sleep(delay)


async def _grant(
    operation: str, payload: dict[str, str], *, client_id: str, client_secret: str
) -> dict[str, Any]:
    """A token response with the fields normalized, or an error."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        status, body = await _request(
            operation,
            "POST",
            TOKEN_URL,
            session,
            json=payload,
            headers=_app_headers(client_id, client_secret),
        )
    if not isinstance(body, dict):
        raise NotionError(operation, f"http_{status}")
    error = body.get("error")
    if error or status != 200:
        raise NotionError(operation, _code(error, "oauth_error") if error else f"http_{status}")
    access_token = _text(body.get("access_token"))
    if access_token is None:
        raise NotionError(operation, "no_access_token")
    expires_in = body.get("expires_in")
    return {
        **body,
        "access_token": access_token,
        "refresh_token": _text(body.get("refresh_token")),
        # ``True`` is an int in Python; a lifetime is a positive whole number of seconds.
        "expires_in": expires_in
        if isinstance(expires_in, int)
        and not isinstance(expires_in, bool)
        and 0 < expires_in <= _MAX_EXPIRES_IN
        else None,
    }


async def exchange_code(
    code: str, *, client_id: str, client_secret: str, redirect_uri: str
) -> dict[str, Any]:
    """Exchange an authorization code. ``workspace_id`` names the account the token is for."""
    grant = await _grant(
        "code exchange",
        {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
        client_id=client_id,
        client_secret=client_secret,
    )
    if not _text(grant.get("workspace_id")):
        raise NotionError("code exchange", "no_workspace_id")
    return grant


async def refresh_access_token(
    refresh_token: str, *, client_id: str, client_secret: str
) -> dict[str, Any]:
    """A new access token and, because Notion rotates, a new refresh token."""
    return await _grant(
        "token refresh",
        {"grant_type": "refresh_token", "refresh_token": refresh_token},
        client_id=client_id,
        client_secret=client_secret,
    )


async def revoke_token(token: str, *, client_id: str, client_secret: str) -> None:
    """Revoke a token at Notion. Only the HTTP status is looked at."""
    async with (
        aiohttp.ClientSession(timeout=TIMEOUT) as session,
        session.post(
            REVOKE_URL,
            json={"token": token},
            headers=_app_headers(client_id, client_secret),
        ) as response,
    ):
        if response.status != 200:
            logger.warning("Notion token revoke returned HTTP %s", response.status)


def _title(item: dict[str, Any]) -> str:
    """A page or data source title as plain text; empty when it has none.

    A page keeps its title in whichever property has ``type == "title"``; a data
    source keeps it in a top-level rich-text ``title`` array.
    """
    title = item.get("title")
    if not isinstance(title, list):
        title = next(
            (
                prop.get("title")
                for prop in (item.get("properties") or {}).values()
                if isinstance(prop, dict) and prop.get("type") == "title"
            ),
            None,
        )
    if not isinstance(title, list):
        return ""
    return "".join(
        part["plain_text"]
        for part in title
        if isinstance(part, dict) and isinstance(part.get("plain_text"), str)
    ).strip()


def _parent(item: dict[str, Any]) -> tuple[str | None, str | None]:
    """``(kind, id)`` of an object's parent, as Notion names the kind (``page_id``, ...)."""
    parent = item.get("parent") if isinstance(item.get("parent"), dict) else {}
    kind = _text(parent.get("type"))
    return kind, (_text(parent.get(kind)) if kind else None)


async def search_shared(access_token: str) -> list[dict[str, Any]]:
    """Every live page and data source the integration was shared, flat.

    Past :data:`MAX_SEARCH_PAGES` pages or :data:`SEARCH_DEADLINE_SECONDS` seconds the
    walk raises ``too_many_results`` or ``timeout`` instead of returning what it has.

    Notion has no "list what this connection can see" call other than search, and
    search returns nested pages next to their parents with no tree. A response
    without a result list, or one that says there is more but gives no cursor, is
    an error and never a prefix: a truncated list would hide roots the user shared.
    An object seen twice is kept once. Archived and trashed objects are dropped,
    since the sync refuses them.
    """
    try:
        return await asyncio.wait_for(_walk_search(access_token), SEARCH_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        raise NotionError("search", "timeout") from None


async def _walk_search(access_token: str) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for _page in range(MAX_SEARCH_PAGES):
            body: dict[str, Any] = {"page_size": SEARCH_PAGE_SIZE}
            if cursor:
                body["start_cursor"] = cursor
            status, data = await _request(
                "search", "POST", SEARCH_URL, session, json=body, headers=_bearer(access_token)
            )
            if not isinstance(data, dict):
                raise NotionError("search", f"http_{status}")
            if status != 200:
                raise NotionError("search", _code(data.get("code"), f"http_{status}"))
            results = data.get("results")
            if not isinstance(results, list):
                raise NotionError("search", "no_results")
            for hit in results:
                kind = {"page": "page", "data_source": "data_source"}.get(
                    hit.get("object") if isinstance(hit, dict) else None
                )
                object_id = _text(hit.get("id")) if kind else None
                if object_id is None or hit.get("archived") is True or hit.get("in_trash") is True:
                    continue
                parent_kind, parent_id = _parent(hit)
                found.setdefault(
                    canonical_id(object_id),
                    {
                        "id": object_id,
                        "kind": kind,
                        "title": _title(hit),
                        "parent_kind": parent_kind,
                        "parent_id": parent_id,
                    },
                )
            cursor = _text(data.get("next_cursor"))
            if data.get("has_more") is not True:
                return list(found.values())
            if cursor is None:
                raise NotionError("search", "no_cursor")
    raise NotionError("search", "too_many_results")


async def verify_token(access_token: str) -> None:
    """Raise :class:`NotionError` unless Notion still accepts this token.

    ``GET /users/me`` is the cheapest authenticated call. A token stored with no expiry
    is otherwise handed to every sync until the end of time, so a workspace that removed
    the integration at Notion would stay connected here with every sync failing.
    """
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        status, data = await _request(
            "token check", "GET", USERS_ME_URL, session, headers=_bearer(access_token)
        )
    if status != 200:
        code = _code(data.get("code") if isinstance(data, dict) else None, f"http_{status}")
        raise NotionError("token check", code)


async def get_parent(access_token: str, kind: str, object_id: str) -> tuple[str, str] | None:
    """``(kind, id)`` of an object's parent, or ``None`` when it sits at the workspace top.

    ``kind`` is ``page``, ``data_source``, ``database`` or ``block``. A parent that
    is a data source row's data source, a database's page, a block's page and so on
    comes back as the kind to fetch next. Raises :class:`NotionError` when the
    object cannot be read, so the caller can tell "not shared" (``object_not_found``,
    ``http_403``) from a real failure.
    """
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        status, data = await _request(
            "parent lookup",
            "GET",
            f"{API_BASE}/{_OBJECT_PATH[kind]}/{object_id}",
            session,
            headers=_bearer(access_token),
        )
    if not isinstance(data, dict):
        raise NotionError("parent lookup", f"http_{status}")
    if status != 200:
        raise NotionError("parent lookup", _code(data.get("code"), f"http_{status}"))
    parent_kind, parent_id = _parent(data)
    next_kind = _PARENT_KINDS.get(parent_kind or "")
    # A workspace parent, or a kind this version does not name, ends the walk up.
    return (next_kind, parent_id) if next_kind and parent_id else None
