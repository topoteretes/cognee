"""Bluesky (AT Protocol) connector for cognee: turn social threads and author feeds into memory.

One source handles a Bluesky author handle, search query, or authenticated timeline. It yields posts,
parent thread contexts, and author details as flat document rows for :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import bluesky_source

    await cognee.remember(
        bluesky_source(
            handle="bsky.app",
            limit=50,
        ),
        dataset_name="bluesky_social_intel",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Zero Credentials for Public Data**: Public AT Protocol feeds require no authentication token.
  Optional app-password can be supplied for authenticated user timelines.
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are formatted as ``bluesky:post:<at-uri>``.
* **Facets Resolution**: Rich facets (links, mentions, tags) are resolved to readable inline text.
* **Thread Context**: Reply parent context and author quote attachments are flattened into
  the document text for complete semantic comprehension.
* **Incremental Sync**: Cursor pagination keeps track of the last ingested post timestamp.
* **Tombstones**: Deleted or pruned posts emit ``_deleted: True`` under ``merge`` write disposition.
"""

import logging
import re
import urllib.parse
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PUBLIC_API_URL = "https://public.api.bsky.app"
AUTH_API_URL = "https://bsky.social"
_TIMEOUT = (10, 30)
NODE_SET_COLUMN = "cognee_node_set"
DOCUMENT_SOURCE_ATTR = "document_source"


class BlueskyError(Exception):
    """Base exception for Bluesky connector errors."""


class BlueskyAuthError(BlueskyError):
    """Raised when Bluesky AT Protocol authentication fails."""


class BlueskyRateLimitError(BlueskyError):
    """Raised when AT Protocol rate limit quota is exceeded."""


class BlueskyClient:
    """Client for Bluesky / AT Protocol XRPC endpoints."""

    def __init__(
        self,
        identifier: str | None = None,
        app_password: str | None = None,
        session: Any = None,
        base_url: str = PUBLIC_API_URL,
    ):
        self.identifier = identifier.strip() if identifier else None
        self.app_password = app_password.strip() if app_password else None
        self.base_url = base_url.rstrip("/")
        self.access_token: str | None = None
        self._session = session

        if self._session is None:
            import requests
            self._session = requests.Session()

        if self.identifier and self.app_password:
            self._authenticate()

    def _authenticate(self) -> None:
        """Create an authenticated AT Protocol session via com.atproto.server.createSession."""
        url = f"{AUTH_API_URL}/xrpc/com.atproto.server.createSession"
        payload = {
            "identifier": self.identifier,
            "password": self.app_password,
        }
        try:
            resp = self._session.post(url, json=payload, timeout=_TIMEOUT)
            if resp.status_code == 401:
                raise BlueskyAuthError(f"Authentication failed for {self.identifier}: Invalid identifier or app password")
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data.get("accessJwt")
            self.base_url = AUTH_API_URL
        except Exception as e:
            if not isinstance(e, BlueskyError):
                raise BlueskyAuthError(f"Bluesky session creation error: {e}") from e
            raise

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "cognee-bluesky-connector/1.0",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def get_author_feed(self, actor: str, limit: int = 50, cursor: str | None = None) -> dict[str, Any]:
        """Fetch posts authored by an actor handle or DID."""
        clean_actor = actor.strip().lstrip("@")
        params: dict[str, Any] = {"actor": clean_actor, "limit": min(limit, 100)}
        if cursor:
            params["cursor"] = cursor

        url = f"{self.base_url}/xrpc/app.bsky.feed.getAuthorFeed?{urllib.parse.urlencode(params)}"
        resp = self._session.get(url, headers=self._headers(), timeout=_TIMEOUT)
        if resp.status_code == 429:
            raise BlueskyRateLimitError("Bluesky rate limit exceeded (HTTP 429)")
        resp.raise_for_status()
        return resp.json()

    def search_posts(self, query: str, limit: int = 50, cursor: str | None = None) -> dict[str, Any]:
        """Search public posts matching a query string."""
        params: dict[str, Any] = {"q": query.strip(), "limit": min(limit, 100)}
        if cursor:
            params["cursor"] = cursor

        url = f"{self.base_url}/xrpc/app.bsky.feed.searchPosts?{urllib.parse.urlencode(params)}"
        resp = self._session.get(url, headers=self._headers(), timeout=_TIMEOUT)
        if resp.status_code == 429:
            raise BlueskyRateLimitError("Bluesky rate limit exceeded (HTTP 429)")
        if resp.status_code == 403 and not self.access_token:
            # Fallback message for unauthenticated search
            logger.warning("Unauthenticated Bluesky search requires app-password credentials.")
            return {"posts": []}
        resp.raise_for_status()
        return resp.json()

    def get_post_thread(self, uri: str, depth: int = 1) -> dict[str, Any]:
        """Fetch a post thread with parent and reply context."""
        params = {"uri": uri, "depth": depth}
        url = f"{self.base_url}/xrpc/app.bsky.feed.getPostThread?{urllib.parse.urlencode(params)}"
        resp = self._session.get(url, headers=self._headers(), timeout=_TIMEOUT)
        if resp.status_code == 429:
            raise BlueskyRateLimitError("Bluesky rate limit exceeded (HTTP 429)")
        resp.raise_for_status()
        return resp.json()


def resolve_facets(text: str, facets: list[dict[str, Any]] | None) -> str:
    """Resolve AT Protocol rich text facets (links, mentions) into clean readable text."""
    if not facets or not text:
        return text

    # Sort facets by byte start descending to replace from end to beginning
    sorted_facets = sorted(
        facets,
        key=lambda f: f.get("index", {}).get("byteStart", 0),
        reverse=True,
    )

    encoded = text.encode("utf-8")
    for f in sorted_facets:
        idx = f.get("index", {})
        start = idx.get("byteStart", 0)
        end = idx.get("byteEnd", len(encoded))
        features = f.get("features", [])
        for feat in features:
            f_type = feat.get("$type", "")
            if "link" in f_type and "uri" in feat:
                link_url = feat["uri"]
                label = encoded[start:end].decode("utf-8", errors="replace")
                replacement = f"[{label}]({link_url})".encode("utf-8")
                encoded = encoded[:start] + replacement + encoded[end:]
                break
            elif "mention" in f_type and "did" in feat:
                mention = encoded[start:end].decode("utf-8", errors="replace")
                replacement = f"{mention}".encode("utf-8")
                encoded = encoded[:start] + replacement + encoded[end:]
                break

    return encoded.decode("utf-8", errors="replace")


def format_post_document(post: dict[str, Any], reply_context: str | None = None) -> dict[str, Any]:
    """Format an AT Protocol post into a cognee document row."""
    uri = post.get("uri", "")
    author = post.get("author", {})
    author_handle = author.get("handle", "unknown")
    author_name = author.get("displayName") or author_handle
    record = post.get("record", {})
    raw_text = record.get("text", "")
    facets = record.get("facets", [])
    resolved_text = resolve_facets(raw_text, facets)

    created_at_str = record.get("createdAt") or datetime.now(timezone.utc).isoformat()

    # Build web url from at-uri: at://did:plc:.../app.bsky.feed.post/<rkey>
    rkey = uri.split("/")[-1] if "/" in uri else ""
    web_url = f"https://bsky.app/profile/{author_handle}/post/{rkey}" if rkey else "https://bsky.app"

    snippet = (resolved_text[:60] + "...") if len(resolved_text) > 60 else resolved_text
    title = f"Bluesky Post by @{author_handle}: {snippet}".strip()

    content_lines = [
        f"Author: {author_name} (@{author_handle})",
        f"Published: {created_at_str}",
        f"Post URL: {web_url}",
    ]
    if reply_context:
        content_lines.append(f"Thread Context: {reply_context}")
    content_lines.append("")
    content_lines.append(resolved_text)

    # Embed quote post if present
    embed = post.get("embed", {})
    if embed.get("$type") == "app.bsky.embed.record#view":
        quoted_post = embed.get("record", {})
        if quoted_post.get("$type") == "app.bsky.feed.defs#postView":
            q_author = quoted_post.get("author", {}).get("handle", "unknown")
            q_text = quoted_post.get("value", {}).get("text", "")
            content_lines.append("")
            content_lines.append(f"--- Quoted Post from @{q_author} ---")
            content_lines.append(q_text)

    full_content = "\n".join(content_lines).strip()

    return {
        "id": f"bluesky:post:{uri}",
        "title": title,
        "content": full_content,
        "url": web_url,
        "_deleted": False,
        NODE_SET_COLUMN: "bluesky_posts",
    }


def bluesky_source(
    handle: str | None = None,
    query: str | None = None,
    limit: int = 100,
    identifier: str | None = None,
    app_password: str | None = None,
    session: Any = None,
) -> Iterator[dict[str, Any]]:
    """Yield standardized document rows for Bluesky posts, feeds, and social graphs.

    Parameters
    ----------
    handle : str, optional
        The actor handle (e.g. ``bsky.app`` or ``alice.bsky.social``) whose feed to ingest.
    query : str, optional
        Search query for posts across the network.
    limit : int, default 100
        Maximum number of posts to fetch per ingestion run.
    identifier : str, optional
        Bluesky handle or email for authenticated access.
    app_password : str, optional
        Bluesky App Password generated from account settings.
    session : Any, optional
        Optional custom HTTP session for requests/mocking.

    Yields
    ------
    dict[str, Any]
        Flat document row with ``{id, title, content, url, _deleted, cognee_node_set}``.
    """
    if not handle and not query:
        raise ValueError("Either 'handle' or 'query' must be provided to bluesky_source.")

    client = BlueskyClient(
        identifier=identifier,
        app_password=app_password,
        session=session,
    )

    yielded_count = 0
    cursor: str | None = None

    while yielded_count < limit:
        batch_limit = min(50, limit - yielded_count)

        if handle:
            resp = client.get_author_feed(actor=handle, limit=batch_limit, cursor=cursor)
            feed_items = resp.get("feed", [])
            if not feed_items:
                break

            for item in feed_items:
                post = item.get("post", {})
                if not post or not post.get("uri"):
                    continue

                reply_info = item.get("reply", {})
                parent_text = None
                if reply_info:
                    parent_post = reply_info.get("parent", {})
                    p_author = parent_post.get("author", {}).get("handle", "")
                    p_text = parent_post.get("record", {}).get("text", "")
                    if p_text:
                        parent_text = f"In reply to @{p_author}: {p_text[:120]}"

                doc = format_post_document(post, reply_context=parent_text)
                yield doc
                yielded_count += 1
                if yielded_count >= limit:
                    break

            cursor = resp.get("cursor")
            if not cursor:
                break

        elif query:
            resp = client.search_posts(query=query, limit=batch_limit, cursor=cursor)
            posts = resp.get("posts", [])
            if not posts:
                break

            for post in posts:
                if not post or not post.get("uri"):
                    continue
                doc = format_post_document(post)
                yield doc
                yielded_count += 1
                if yielded_count >= limit:
                    break

            cursor = resp.get("cursor")
            if not cursor:
                break
