"""Linkding connector for cognee: a ``dlt`` source that turns self-hosted bookmarks into memory.

Linkding is an open-source, self-hosted bookmark manager. This connector connects to a Linkding
instance using its REST API, ingesting bookmarks with their titles, URLs, descriptions,
notes, and tags into Cognee memory. It supports incremental sync via `modified_since` timestamps,
node sets scoped by tag, and deletion tracking via 404 status checks.
"""

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from typing import Any

import requests

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = get_logger("linkding_connector")

DEFAULT_PAGE_SIZE = 100
ID_SCAN_PAGE_SIZE = 1000
_TIMEOUT = (10, 30)


class LinkdingError(RuntimeError):
    """Base error for Linkding connector."""


class LinkdingAuthError(LinkdingError):
    """Raised when the Linkding API token is rejected (401 or 403)."""


class LinkdingConnectionError(LinkdingError):
    """Raised when connecting to the Linkding instance fails."""


class LinkdingClient:
    """HTTP client for the Linkding REST API."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {api_token}",
                "Accept": "application/json",
                "User-Agent": "cognee-linkding-connector/1.0",
            }
        )

    def request(
        self,
        method: str,
        endpoint_or_url: str,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = (
            endpoint_or_url
            if endpoint_or_url.startswith(("http://", "https://"))
            else f"{self.base_url}{endpoint_or_url}"
        )
        try:
            resp = self.session.request(
                method,
                url,
                params=params,
                verify=self.verify_ssl,
                timeout=_TIMEOUT,
            )
        except requests.RequestException as err:
            raise LinkdingConnectionError(f"Failed to connect to Linkding at {url}: {err}") from err

        if resp.status_code in (401, 403):
            raise LinkdingAuthError("Invalid Linkding API token or unauthorized access.")
        return resp


def _bookmark_id(raw_id: int | str) -> str:
    return f"linkding:bookmark:{raw_id}"


def _format_content(bookmark: dict[str, Any]) -> str:
    """Format bookmark attributes into structured Markdown."""
    title = (
        bookmark.get("title")
        or bookmark.get("website_title")
        or bookmark.get("url")
        or "Untitled Bookmark"
    )
    url = bookmark.get("url", "")
    description = bookmark.get("description") or bookmark.get("website_description") or ""
    notes = bookmark.get("notes") or ""
    tags = bookmark.get("tag_names") or []
    is_archived = bookmark.get("is_archived", False)
    unread = bookmark.get("unread", False)

    lines = [f"# {title}", f"**URL:** {url}"]
    if description:
        lines.append(f"**Description:** {description}")
    if notes:
        lines.extend(["**Notes:**", notes])
    if tags:
        lines.append(f"**Tags:** {', '.join(sorted(tags))}")

    status_parts = ["Archived" if is_archived else "Active"]
    if unread:
        status_parts.append("Unread")
    lines.append(f"**Status:** {', '.join(status_parts)}")

    return "\n\n".join(lines).strip()


def _bookmark_row(bookmark: dict[str, Any]) -> dict[str, Any]:
    """Format document row with ID, title, content, URL, and tag node sets."""
    b_id = bookmark["id"]
    title = (
        bookmark.get("title")
        or bookmark.get("website_title")
        or bookmark.get("url")
        or "Untitled Bookmark"
    )
    url = bookmark.get("url", "")
    tags = bookmark.get("tag_names") or []

    row: dict[str, Any] = {
        "id": _bookmark_id(b_id),
        "title": title,
        "content": _format_content(bookmark),
        "url": url,
        "_deleted": False,
    }

    node_set = ["linkding:bookmark"]
    for tag in sorted(tags):
        clean_tag = re.sub(r"[^\w\s-]", "", tag).strip()
        if clean_tag:
            node_set.append(f"linkding:tag:{clean_tag}")
    row[NODE_SET_COLUMN] = node_set

    return row


def _fetch_paged_results(
    client: LinkdingClient,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all pages from a paginated Linkding endpoint."""
    results: list[dict[str, Any]] = []
    next_url: str | None = endpoint
    current_params = params

    while next_url:
        resp = client.request("GET", next_url, params=current_params)
        if resp.status_code != 200:
            raise LinkdingError(
                f"Linkding API returned unexpected HTTP {resp.status_code} for {next_url}"
            )
        data = resp.json()
        results.extend(data.get("results", []))
        next_url = data.get("next")
        current_params = None  # query params are already encoded in next_url

    return results


def _iter_rows(
    client: LinkdingClient,
    state: dict[str, Any],
    stats: dict[str, int],
    include_archived: bool = False,
    bundle: str | int | None = None,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield bookmark rows and tombstone records."""
    cursor: str | None = state.get("cursor")
    previous_members: set[int] = set(state.get("members", []))
    highest_modified: str | None = cursor

    endpoints = ["/api/bookmarks/"]
    if include_archived:
        endpoints.append("/api/bookmarks/archived/")

    # 1. Fetch changed or new bookmarks
    query_params: dict[str, Any] = {"limit": DEFAULT_PAGE_SIZE}
    if cursor:
        query_params["modified_since"] = cursor
    if bundle is not None:
        query_params["bundle"] = bundle

    synced_ids: set[int] = set()

    for endpoint in endpoints:
        bookmarks = _fetch_paged_results(client, endpoint, params=query_params)
        for bm in bookmarks:
            if check_active is not None:
                check_active()

            b_id = bm["id"]
            synced_ids.add(b_id)

            date_mod = bm.get("date_modified") or bm.get("date_added")
            if date_mod and (highest_modified is None or date_mod > highest_modified):
                highest_modified = date_mod

            yield _bookmark_row(bm)
            stats["synced"] += 1

    # 2. Collect all active IDs currently in scope to verify deletions
    current_alive_ids: set[int] = set()
    scan_params: dict[str, Any] = {"limit": ID_SCAN_PAGE_SIZE}
    if bundle is not None:
        scan_params["bundle"] = bundle

    for endpoint in endpoints:
        all_results = _fetch_paged_results(client, endpoint, params=scan_params)
        for item in all_results:
            current_alive_ids.add(item["id"])

    # 3. Check potentially deleted bookmarks
    # If an ID was in previous_members but is no longer in current_alive_ids, check if 404
    missing_ids = previous_members - current_alive_ids
    for m_id in sorted(missing_ids):
        if check_active is not None:
            check_active()

        check_resp = client.request("GET", f"/api/bookmarks/{m_id}/")
        if check_resp.status_code == 404:
            yield {"id": _bookmark_id(m_id), "_deleted": True}
            stats["deleted"] += 1
        elif check_resp.status_code == 200:
            # Still exists (e.g. archived when include_archived is False), keep in members
            current_alive_ids.add(m_id)

    # 4. Update sync state
    state.update(
        v=1,
        cursor=highest_modified,
        members=sorted(current_alive_ids),
    )
    logger.info(
        "Linkding sync finished: %d bookmark(s) synced, %d deleted.",
        stats["synced"],
        stats["deleted"],
    )


def linkding_source(
    *,
    base_url: str,
    api_token: str,
    resource_name: str | None = None,
    include_archived: bool = False,
    bundle: str | int | None = None,
    verify_ssl: bool = True,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields bookmarks from a self-hosted Linkding instance.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``.

    Args:
        base_url: URL of the Linkding instance (e.g. ``http://localhost:9090``).
        api_token: REST API token from Linkding Settings -> Integrations.
        resource_name: Unique resource identifier for dlt state. Defaults to a hash of the base URL.
        include_archived: Whether to include archived bookmarks in addition to active ones.
        bundle: Filter bookmarks by Linkding bundle ID or name.
        verify_ssl: Whether to verify SSL certificates (set False for self-signed certificates).
        check_active: Host authorization checkpoint callback.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Linkding sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string.")
    if not isinstance(api_token, str) or not api_token.strip():
        raise ValueError("api_token must be a non-empty string.")

    norm_url = base_url.strip().rstrip("/")
    if resource_name is None:
        resource_name = f"linkding_{hashlib.sha256(norm_url.encode()).hexdigest()[:12]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")

    client = LinkdingClient(base_url=norm_url, api_token=api_token.strip(), verify_ssl=verify_ssl)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def linkding_bookmarks():
        stats.update(synced=0, deleted=0, failed=0)
        rows = _iter_rows(
            client=client,
            state=dlt.current.resource_state(),
            stats=stats,
            include_archived=include_archived,
            bundle=bundle,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = linkding_bookmarks()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "linkding")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
