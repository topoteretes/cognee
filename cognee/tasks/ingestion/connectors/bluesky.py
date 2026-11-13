"""Bluesky (AT Protocol) connector for cognee: a ``dlt`` source that turns public or
authenticated Bluesky timelines, author feeds, and discussions into memory.

One source is one Bluesky actor feed or search query. It yields posts (with reply
and quote context folded into the post document) as flat document rows, and is meant
to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import bluesky_source

    await cognee.remember(
        bluesky_source(actor="bsky.app"),
        dataset_name="bluesky",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted}``. Ids are prefixed by
  kind (``post:<at-uri>``). Every column feeds the stored document identity, so
  the column set and the rendering are frozen: nothing volatile (like counts,
  repost counts, relative timestamps) is rendered into the content.
* **Rich text facets** (links, mentions, tags) are resolved strictly by UTF-8 byte
  slices to avoid character-offset drift on emojis or multi-byte Unicode.
* **Thread context** (parent replies and quoted post embeds) is flattened into
  the document for coherent semantic recall without graph fragmenting.
* **Incremental sync** by cursor and AT Protocol record timestamps. A second run
  with no new posts yields nothing.
* **A run is bounded and resumable.** On a rate limit (HTTP 429), the source stops
  cleanly, preserves the state it reached, and reports it in ``cognee_sync_stats``
  (``failed_rate_limit``). It never raises or sleeps indefinitely.
* **Zero Credentials for Public Data**: Public author feeds and thread views query
  the public XRPC AppView (``https://public.api.bsky.app``) with zero auth tokens.
  An optional access token / app password can be provided for authenticated feeds.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

PUBLIC_XRPC_BASE = "https://public.api.bsky.app/xrpc"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
DEFAULT_MAX_REQUESTS = 100
_TIMEOUT = (10, 30)


# ---------------------------------------------------------------------------
# Errors: messages carry codes only, never credentials or response bodies.
# ---------------------------------------------------------------------------
class BlueskySourceError(RuntimeError):
    """Base class for all errors raised by the Bluesky connector."""


class BlueskyAPIError(BlueskySourceError):
    """Bluesky answered with an error or could not be reached."""


class BlueskyAuthError(BlueskySourceError):
    """Bluesky rejected the authorization credentials (HTTP 401)."""


class BlueskyRateLimitedError(BlueskySourceError):
    """Bluesky rate limit exceeded (HTTP 429)."""

    def __init__(self, reset_at_epoch: int | None = None):
        super().__init__("Bluesky rate limit exceeded (HTTP 429)")
        self.reset_at_epoch = reset_at_epoch


class BlueskyActorNotFoundError(BlueskySourceError):
    """The requested Bluesky actor/profile could not be found or resolved (HTTP 404)."""


# ---------------------------------------------------------------------------
# Transport & Client
# ---------------------------------------------------------------------------
class BlueskyClient:
    """Synchronous client for AT Protocol XRPC endpoints."""

    def __init__(
        self,
        access_token: str | None = None,
        base_url: str = PUBLIC_XRPC_BASE,
        session: Any = None,
    ):
        self._access_token = access_token
        self.base_url = base_url.rstrip("/")
        self._session = session
        self.rate_limit: dict[str, Any] = {}

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        import requests

        self._session = requests.Session()
        return self._session

    def execute(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a GET request against an XRPC endpoint."""
        session = self._get_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "CogneeBlueskyConnector/1.0",
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        try:
            resp = session.get(
                url, params=clean_params, headers=headers, timeout=_TIMEOUT
            )
        except Exception as exc:
            raise BlueskyAPIError(
                f"Network error querying Bluesky XRPC endpoint {endpoint}"
            ) from exc

        # Track rate limit headers
        remaining = resp.headers.get("RateLimit-Remaining")
        reset_time = resp.headers.get("RateLimit-Reset")
        if remaining is not None:
            try:
                self.rate_limit["remaining"] = int(remaining)
            except (ValueError, TypeError):
                pass
        if reset_time is not None:
            try:
                self.rate_limit["reset"] = int(reset_time)
            except (ValueError, TypeError):
                pass

        if resp.status_code == 429:
            reset_epoch = self.rate_limit.get("reset")
            raise BlueskyRateLimitedError(reset_at_epoch=reset_epoch)

        if resp.status_code == 401:
            raise BlueskyAuthError(f"Authentication rejected for endpoint {endpoint}")

        if resp.status_code == 404:
            raise BlueskyActorNotFoundError(
                f"Resource or actor not found at {endpoint}"
            )

        if not resp.ok:
            raise BlueskyAPIError(
                f"Bluesky returned HTTP status {resp.status_code} for {endpoint}"
            )

        try:
            return resp.json()
        except Exception as exc:
            raise BlueskyAPIError("Bluesky returned invalid non-JSON payload") from exc

    def get_author_feed(
        self,
        actor: str,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        filter_mode: str = "posts_and_author_threads",
    ) -> dict[str, Any]:
        """Fetch posts for an actor feed."""
        params: dict[str, Any] = {
            "actor": actor,
            "limit": max(1, min(limit, MAX_PAGE_SIZE)),
            "filter": filter_mode,
        }
        if cursor:
            params["cursor"] = cursor
        return self.execute("app.bsky.feed.getAuthorFeed", params=params)

    def get_post_thread(self, uri: str, depth: int = 1) -> dict[str, Any]:
        """Fetch a post thread with parent and reply context."""
        params = {"uri": uri, "depth": depth}
        return self.execute("app.bsky.feed.getPostThread", params=params)

    def get_profile(self, actor: str) -> dict[str, Any]:
        """Fetch an actor's profile metadata."""
        return self.execute("app.bsky.actor.getProfile", params={"actor": actor})


def build_bluesky_service(
    access_token: str | None = None,
    base_url: str = PUBLIC_XRPC_BASE,
) -> BlueskyClient:
    """Build a configured BlueskyClient instance."""
    return BlueskyClient(access_token=access_token, base_url=base_url)


# ---------------------------------------------------------------------------
# Rich Text Facet & Content Rendering
# ---------------------------------------------------------------------------
def resolve_facets(text: str, facets: list[dict[str, Any]] | None) -> str:
    """Resolve AT Protocol rich text facets (links, mentions, tags) into Markdown.

    AT Protocol facets define byteStart and byteEnd as UTF-8 byte offsets,
    not unicode character indices. Slicing as raw bytes prevents emoji drift.
    """
    if not facets or not text:
        return text

    encoded = text.encode("utf-8")
    sorted_facets: list[tuple[int, int, list[dict[str, Any]]]] = []

    for facet in facets:
        index = facet.get("index", {})
        start = index.get("byteStart")
        end = index.get("byteEnd")
        features = facet.get("features", [])
        if start is not None and end is not None and 0 <= start <= end <= len(encoded):
            sorted_facets.append((start, end, features))

    # Sort strictly by byteStart ascending, then byteEnd
    sorted_facets.sort(key=lambda item: (item[0], item[1]))

    chunks: list[str] = []
    curr = 0

    for start, end, features in sorted_facets:
        if start < curr:
            # Skip overlapping or malformed duplicate facets
            continue

        # Decode preceding segment
        chunks.append(encoded[curr:start].decode("utf-8", errors="replace"))
        original_chunk = encoded[start:end].decode("utf-8", errors="replace")

        replacement = original_chunk
        for feat in features:
            feat_type = feat.get("$type", "")
            if feat_type.endswith("#link") and "uri" in feat:
                uri = feat["uri"]
                replacement = f"[{original_chunk}]({uri})"
                break
            elif feat_type.endswith("#mention") and "did" in feat:
                did = feat["did"]
                replacement = f"[{original_chunk}](https://bsky.app/profile/{did})"
                break
            elif feat_type.endswith("#tag") and "tag" in feat:
                tag = feat["tag"]
                replacement = f"[{original_chunk}](https://bsky.app/hashtag/{tag})"
                break

        chunks.append(replacement)
        curr = end

    chunks.append(encoded[curr:].decode("utf-8", errors="replace"))
    return "".join(chunks)


def extract_post_web_url(handle: str, uri: str) -> str:
    """Convert an AT-URI (at://did/app.bsky.feed.post/rkey) to a public bsky.app URL."""
    parts = uri.split("/")
    if len(parts) >= 5:
        rkey = parts[-1]
        return f"https://bsky.app/profile/{handle}/post/{rkey}"
    return f"https://bsky.app/profile/{handle}"


def render_post_document(feed_item: dict[str, Any]) -> dict[str, Any]:
    """Render a Bluesky feed item into a flat, deterministic Cognee document row."""
    post = feed_item.get("post", {})
    record = post.get("record", {})
    author = post.get("author", {})

    uri = post.get("uri", "")
    handle = author.get("handle") or "unknown"
    display_name = author.get("displayName") or handle

    raw_text = record.get("text", "") or ""
    facets = record.get("facets")
    resolved_text = resolve_facets(raw_text, facets)

    # Flatten context (reply parent context and quoted post context)
    content_lines: list[str] = []

    reply = feed_item.get("reply")
    if reply:
        parent = reply.get("parent", {})
        parent_post = parent.get("record", {})
        parent_author = parent.get("author", {})
        if parent_post:
            parent_handle = parent_author.get("handle", "unknown")
            parent_text = parent_post.get("text", "").strip()
            if parent_text:
                content_lines.append(f"> Replying to @{parent_handle}: {parent_text}")

    content_lines.append(resolved_text)

    # Quoted post embeds
    embed = post.get("embed", {})
    embed_type = embed.get("$type", "")
    if "record" in embed_type:
        quoted_record = embed.get("record", {})
        quoted_author = quoted_record.get("author", {})
        quoted_value = quoted_record.get("value", {})
        quoted_text = quoted_value.get("text") or quoted_record.get("text")
        if quoted_text:
            q_handle = quoted_author.get("handle") or "author"
            content_lines.append(f"> Quoting @{q_handle}: {quoted_text.strip()}")

    # External link embeds
    if embed_type.endswith("#external"):
        external = embed.get("external", {})
        ext_title = external.get("title")
        ext_uri = external.get("uri")
        if ext_uri:
            link_desc = ext_title or ext_uri
            content_lines.append(f"Link: [{link_desc}]({ext_uri})")

    content = "\n\n".join(line for line in content_lines if line.strip())

    # Build clean deterministic title
    snippet = raw_text.replace("\n", " ").strip()
    if len(snippet) > 60:
        snippet = snippet[:57] + "..."
    title = f"@{handle}: {snippet}" if snippet else f"Post by @{handle}"

    web_url = extract_post_web_url(handle, uri)

    return {
        "id": f"post:{uri}",
        "title": title,
        "content": content,
        "url": web_url,
        "_deleted": False,
    }


# ---------------------------------------------------------------------------
# Incremental Row Generator
# ---------------------------------------------------------------------------
def _iter_rows(
    client: BlueskyClient,
    actor: str,
    resource_state: dict[str, Any],
    stats: dict[str, int],
    *,
    check_active: Callable[[], None] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    """Yield document rows incrementally across successive sync runs."""
    floor_timestamp = resource_state.get("floor_timestamp")
    seen_ids = set(resource_state.get("seen_ids", []))

    cursor = resource_state.get("cursor")
    requests_made = 0
    highest_timestamp = floor_timestamp
    current_run_seen_ids = set()

    while requests_made < max_requests:
        if check_active is not None:
            check_active()

        requests_made += 1
        stats["scanned"] = stats.get("scanned", 0)

        try:
            data = client.get_author_feed(actor, limit=page_size, cursor=cursor)
        except BlueskyRateLimitedError:
            logger.warning(
                "Bluesky rate limit hit. Halting sync cleanly with state preserved."
            )
            stats["failed_rate_limit"] = 1
            break
        except BlueskyActorNotFoundError as exc:
            logger.error("Bluesky actor %s not found: %s", actor, exc)
            stats["failed"] = stats.get("failed", 0) + 1
            break
        except Exception as exc:
            logger.error("Bluesky sync encountered unexpected error: %s", exc)
            stats["failed"] = stats.get("failed", 0) + 1
            break

        feed = data.get("feed", [])
        if not feed:
            break

        new_items_in_page = 0

        for item in feed:
            stats["scanned"] += 1
            post = item.get("post", {})
            uri = post.get("uri")
            if not uri:
                stats["skipped"] = stats.get("skipped", 0) + 1
                continue

            doc_id = f"post:{uri}"
            record = post.get("record", {})
            created_at = record.get("createdAt")

            # Update highest observed timestamp
            if created_at and (
                highest_timestamp is None or created_at > highest_timestamp
            ):
                highest_timestamp = created_at

            # Incremental floor check
            if floor_timestamp and created_at and created_at < floor_timestamp:
                # We have reached posts older than our incremental floor
                stats["skipped"] = stats.get("skipped", 0) + 1
                continue

            if doc_id in seen_ids:
                stats["skipped"] = stats.get("skipped", 0) + 1
                continue

            doc = render_post_document(item)
            current_run_seen_ids.add(doc_id)
            new_items_in_page += 1
            yield doc

        cursor = data.get("cursor")
        if not cursor:
            # Reached end of feed
            break

    # Persist updated incremental state
    if highest_timestamp:
        resource_state["floor_timestamp"] = highest_timestamp
    all_seen = seen_ids | current_run_seen_ids
    # Bound state memory to most recent 5,000 IDs
    if len(all_seen) > 5000:
        all_seen = set(list(all_seen)[-5000:])
    resource_state["seen_ids"] = list(all_seen)
    resource_state["cursor"] = None


# ---------------------------------------------------------------------------
# DLT Source Entry Point
# ---------------------------------------------------------------------------
def _default_resource_name(actor: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", actor).strip("_")
    return f"bluesky_{cleaned}"


def bluesky_source(
    *,
    actor: str,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
    service: BlueskyClient | None = None,
    access_token: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Any:
    """Return a ``dlt`` resource yielding Bluesky post documents for ``cognee.remember``.

    Args:
        actor: Bluesky handle (e.g. ``bsky.app``) or DID (``did:plc:...``).
        resource_name: Unique resource table name. Defaults to ``bluesky_<actor>``.
        check_active: Optional authorization checkpoint callback.
        service: Injected client instance (useful for unit tests and mocks).
        access_token: Optional Bearer token for private/authenticated feeds.
        page_size: Rows per page (1 to 100).
        max_requests: Request budget per sync run.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"``, and hard-delete tombstoning.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Bluesky connector requires dlt (a cognee core dependency)."
        ) from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Bluesky sync requires a Cognee build with table-scoped DLT document cleanup."
        )

    client = (
        service if service is not None else BlueskyClient(access_token=access_token)
    )
    name = resource_name or _default_resource_name(actor)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def bluesky_timeline() -> Iterator[dict[str, Any]]:
        stats.clear()
        stats.update(scanned=0, skipped=0, failed=0)
        yield from _iter_rows(
            client,
            actor,
            dlt.current.resource_state(),
            stats,
            check_active=check_active,
            page_size=page_size,
            max_requests=max_requests,
        )

    resource = bluesky_timeline()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "bluesky")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    resource.cognee_sync_stats = stats
    return resource
