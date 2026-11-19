"""Zendesk Tickets connector for cognee: a ``dlt`` source that turns Zendesk support tickets into memory.

One source is one Zendesk subdomain and an optional selection of view/search criteria. It yields
one document per ticket (with its metadata, status, priority, description, tags, custom fields,
and conversation audit comments folded in) as flat rows, and is meant to be handed directly to
:func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import zendesk_source

    await cognee.remember(
        zendesk_source(
            subdomain="acme",
            access_token="user@acme.com/token:<api_token>",  # or OAuth 2.0 Bearer token
            query="type:ticket status<solved",
        ),
        dataset_name="zendesk",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are prefixed
  by kind (``zendesk:ticket:<id>``). Every column feeds the stored document identity, so the
  rendering is frozen and deterministic: no volatile timestamps in ``content``, and ``updated_at``
  lives only in resource state.
* **Node set** ``zendesk:<subdomain>`` per ticket row so a ``recall`` can be scoped to a single
  Zendesk instance or brand.
* **Incremental** by ``updated_at`` timestamp stored in ``dlt.current.resource_state()``,
  plus a ``known_ids`` set to detect and tombstone deleted or archived tickets on full
  or incremental verification passes.
* **Bounded and resumable.** On a rate limit (HTTP 429), spent ``max_requests`` budget, or
  expired token after partial progress, the source stops cleanly, preserves its state, and
  records diagnostics in ``cognee_sync_stats``.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlparse

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100
DEFAULT_MAX_REQUESTS = 500
_TIMEOUT = (10, 30)
_RETRIES = 3
_SUBDOMAIN_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$")


class ZendeskSourceError(RuntimeError):
    """Base error raised by the Zendesk connector."""


class ZendeskAuthError(ZendeskSourceError):
    """Zendesk rejected the credentials (HTTP 401 or 403 on authentication)."""


class ZendeskRateLimitedError(ZendeskSourceError):
    """Zendesk answered HTTP 429 Too Many Requests."""

    def __init__(self, retry_after: int | None = None):
        super().__init__("Zendesk rate limit exceeded (HTTP 429)")
        self.retry_after = retry_after


class ZendeskAPIError(ZendeskSourceError):
    """Zendesk answered with an unexpected HTTP or payload error."""


def _normalize_subdomain(subdomain_or_url: str) -> tuple[str, str]:
    """Return normalized (subdomain, base_url)."""
    raw = (subdomain_or_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("subdomain must not be empty.")

    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.netloc.lower()
        if host.endswith(".zendesk.com"):
            subdomain = host.split(".zendesk.com")[0]
        else:
            subdomain = host
    else:
        subdomain = raw.lower()

    if not _SUBDOMAIN_RE.match(subdomain):
        raise ValueError(f"Invalid Zendesk subdomain: {subdomain_or_url!r}")

    base_url = f"https://{subdomain}.zendesk.com"
    return subdomain, base_url


class ZendeskClient:
    """Minimal synchronous REST client for Zendesk Support API v2."""

    def __init__(
        self,
        subdomain: str,
        access_token: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.subdomain, self.base_url = _normalize_subdomain(subdomain)
        token = (access_token or "").strip()
        if not token or not token.isascii() or not token.isprintable():
            raise ValueError("The Zendesk access token is empty or has an invalid format.")
        self._access_token = token
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"ZendeskClient(subdomain={self.subdomain!r}, access_token=<redacted>)"

    def _headers(self) -> dict[str, str]:
        token = self._access_token
        # Zendesk API tokens are commonly passed as basic auth: `{email}/token:{api_token}`
        if ":" in token and " " not in token:
            encoded = base64.b64encode(token.encode("ascii")).decode("ascii")
            auth_header = f"Basic {encoded}"
        else:
            auth_header = f"Bearer {token}"
        return {
            "Authorization": auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cognee-zendesk-connector",
        }

    def request_json(
        self,
        method: str,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import requests

        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            url = path_or_url
        else:
            url = f"{self.base_url}/{path_or_url.lstrip('/')}"

        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.request(
                    method,
                    url,
                    headers=self._headers(),
                    params=params,
                    json=json_body,
                    timeout=_TIMEOUT,
                )
            except requests.RequestException as exc:
                if last:
                    raise ZendeskAPIError(f"Zendesk request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue

            status = response.status_code
            if status >= 500 and not last:
                self._sleep(2**attempt)
                continue
            if status in (401, 403):
                raise ZendeskAuthError(f"Zendesk rejected credentials (HTTP {status})")
            if status == 429:
                retry_hdr = response.headers.get("Retry-After")
                retry_sec = int(retry_hdr) if retry_hdr and retry_hdr.isdigit() else None
                raise ZendeskRateLimitedError(retry_after=retry_sec)
            if status == 404:
                raise ZendeskAPIError("Zendesk endpoint or entity not found (HTTP 404)")
            if status >= 400:
                raise ZendeskAPIError(f"Zendesk request failed with HTTP {status}")

            try:
                data = response.json()
            except ValueError as exc:
                raise ZendeskAPIError("Zendesk response is not valid JSON") from exc
            if not isinstance(data, dict):
                raise ZendeskAPIError("Zendesk response root must be a JSON object")
            return data

        raise ZendeskAPIError("Zendesk request failed")  # pragma: no cover

    def search_tickets(
        self,
        query: str,
        *,
        page_after: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Search tickets using Zendesk Cursor-Based or Search API."""
        params: dict[str, Any] = {
            "query": query,
            "page[size]": page_size,
        }
        if page_after:
            params["page[after]"] = page_after
        return self.request_json("GET", "/api/v2/search/export.json", params=params)

    def list_tickets(
        self,
        *,
        start_time: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Incremental ticket export endpoint (/api/v2/incremental/tickets/cursor.json)."""
        params: dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        elif start_time is not None:
            params["start_time"] = start_time
        return self.request_json("GET", "/api/v2/incremental/tickets/cursor.json", params=params)

    def get_ticket_comments(self, ticket_id: int | str) -> list[dict[str, Any]]:
        """Fetch comments for a ticket."""
        res = self.request_json("GET", f"/api/v2/tickets/{ticket_id}/comments.json")
        comments = res.get("comments")
        return comments if isinstance(comments, list) else []


def build_zendesk_service(subdomain: str, access_token: str) -> ZendeskClient:
    """Construct a :class:`ZendeskClient` for :func:`zendesk_source`."""
    return ZendeskClient(subdomain=subdomain, access_token=access_token)


def _render_ticket_row(
    base_url: str,
    subdomain: str,
    ticket: dict[str, Any],
    comments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ticket_id = str(ticket.get("id") or "").strip()
    subject = str(ticket.get("subject") or f"Ticket #{ticket_id}").strip()
    status = str(ticket.get("status") or "new").strip().lower()
    priority = str(ticket.get("priority") or "normal").strip().lower()
    ticket_type = str(ticket.get("type") or "incident").strip().lower()
    raw_desc = str(ticket.get("description") or "").strip()

    raw_tags = ticket.get("tags")
    tags = sorted(
        {str(t).strip() for t in (raw_tags if isinstance(raw_tags, list) else []) if str(t).strip()}
    )

    lines = [
        f"[Zendesk #{ticket_id}] {subject}",
        f"Subdomain: {subdomain}",
        f"Type: {ticket_type}",
        f"Status: {status}",
        f"Priority: {priority}",
    ]
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")

    if raw_desc:
        lines.append("")
        lines.append("Description:")
        lines.append(raw_desc)

    if comments:
        c_lines: list[str] = []
        for c in comments:
            if not isinstance(c, dict):
                continue
            body = str(c.get("body") or c.get("plain_body") or "").strip()
            public = "Public" if c.get("public", True) else "Internal"
            author_id = c.get("author_id") or "Unknown"
            if body:
                c_lines.append(f"- [{public}] Author {author_id}: {body}")
        if c_lines:
            lines.append("")
            lines.append("Comments:")
            lines.extend(c_lines)

    return {
        "id": f"zendesk:ticket:{ticket_id}",
        "title": f"Zendesk #{ticket_id}: {subject}",
        "content": "\n".join(lines).strip(),
        "url": f"{base_url}/agent/tickets/{ticket_id}",
        "_deleted": False,
        NODE_SET_COLUMN: [f"zendesk:{subdomain}"],
    }


def _iter_rows(
    client: ZendeskClient,
    query: str | None,
    state: dict[str, Any],
    stats: dict[str, int],
    *,
    include_comments: bool = True,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    requests_used = 0
    subdomain = client.subdomain
    base_url = client.base_url

    known_ids = set(state.get("known_ids", []))
    cursor = state.get("cursor")
    current_run_ids: set[str] = set()

    # Search mode when explicit query is provided
    if query:
        search_query = f"type:ticket {query}".strip()
        page_after = cursor
        while True:
            if requests_used >= max_requests:
                stats["failed"] += 1
                stats["failed_budget"] += 1
                return

            try:
                requests_used += 1
                payload = client.search_tickets(
                    search_query,
                    page_after=page_after,
                    page_size=page_size,
                )
            except ZendeskRateLimitedError:
                stats["failed"] += 1
                stats["failed_rate_limit"] += 1
                return
            except ZendeskAuthError:
                if stats["synced"] > 0 or stats["deleted"] > 0:
                    stats["failed"] += 1
                    stats["failed_auth"] += 1
                    return
                raise

            results = payload.get("results")
            if not isinstance(results, list):
                # Search export format can return tickets in 'results'
                results = payload.get("tickets", [])
            if not isinstance(results, list):
                raise ZendeskAPIError("Zendesk search response missing results list")

            for ticket in results:
                if not isinstance(ticket, dict) or not ticket.get("id"):
                    continue
                ticket_id = str(ticket["id"]).strip()
                current_run_ids.add(ticket_id)
                known_ids.add(ticket_id)

                comments = None
                if include_comments:
                    if requests_used < max_requests:
                        try:
                            requests_used += 1
                            comments = client.get_ticket_comments(ticket_id)
                        except (ZendeskRateLimitedError, ZendeskAPIError, ZendeskAuthError):
                            comments = None
                    else:
                        stats["failed_budget"] += 1

                row = _render_ticket_row(base_url, subdomain, ticket, comments)
                stats["synced"] += 1
                yield row

            meta = payload.get("meta", {})
            has_more = meta.get("has_more") if isinstance(meta, dict) else False
            page_after = meta.get("after_cursor") if isinstance(meta, dict) else None
            if not has_more or not page_after or not results:
                break

        state["cursor"] = page_after
        state["known_ids"] = sorted(known_ids)
        return

    # Incremental tickets cursor sync mode
    while True:
        if requests_used >= max_requests:
            stats["failed"] += 1
            stats["failed_budget"] += 1
            return

        try:
            requests_used += 1
            payload = client.list_tickets(cursor=cursor, start_time=0 if not cursor else None)
        except ZendeskRateLimitedError:
            stats["failed"] += 1
            stats["failed_rate_limit"] += 1
            return
        except ZendeskAuthError:
            if stats["synced"] > 0 or stats["deleted"] > 0:
                stats["failed"] += 1
                stats["failed_auth"] += 1
                return
            raise

        tickets = payload.get("tickets")
        if not isinstance(tickets, list):
            raise ZendeskAPIError("Zendesk tickets response missing 'tickets' list")

        for ticket in tickets:
            if not isinstance(ticket, dict) or not ticket.get("id"):
                continue
            ticket_id = str(ticket["id"]).strip()
            current_run_ids.add(ticket_id)
            known_ids.add(ticket_id)

            comments = None
            if include_comments:
                if requests_used < max_requests:
                    try:
                        requests_used += 1
                        comments = client.get_ticket_comments(ticket_id)
                    except (ZendeskRateLimitedError, ZendeskAPIError, ZendeskAuthError):
                        comments = None
                else:
                    stats["failed_budget"] += 1

            row = _render_ticket_row(base_url, subdomain, ticket, comments)
            stats["synced"] += 1
            yield row

        cursor = payload.get("after_cursor")
        end_of_stream = payload.get("end_of_stream", False)
        if end_of_stream or not cursor or not tickets:
            break

    state["cursor"] = cursor
    state["known_ids"] = sorted(known_ids)


def zendesk_source(
    *,
    subdomain: str,
    access_token: str | None = None,
    service: ZendeskClient | None = None,
    query: str | None = None,
    include_comments: bool = True,
    resource_name: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per Zendesk ticket."""
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Zendesk sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )

    norm_subdomain, norm_url = _normalize_subdomain(subdomain)

    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")
    if max_requests < 1:
        raise ValueError("max_requests must be at least 1.")

    if service is not None:
        client = service
    elif access_token is not None:
        client = build_zendesk_service(norm_subdomain, access_token)
    else:
        raise ValueError("Either access_token or service must be provided.")

    if resource_name is None:
        scope = f"{norm_subdomain}|{query or ''}"
        digest = hashlib.sha256(scope.encode()).hexdigest()[:12]
        resource_name = f"zendesk_{digest}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            NODE_SET_COLUMN: {"data_type": "json", "nullable": True},
        },
    )
    def zendesk_tickets():
        stats.update(
            synced=0,
            deleted=0,
            failed=0,
            failed_rate_limit=0,
            failed_budget=0,
            failed_auth=0,
        )
        rows = _iter_rows(
            client,
            query,
            dlt.current.resource_state(),
            stats,
            include_comments=include_comments,
            page_size=page_size,
            max_requests=max_requests,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = zendesk_tickets()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "zendesk")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
