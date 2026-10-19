"""Mattermost data-source connector for cognee: turn team chat into AI memory.

One source ingests channels and discussion threads from a self-hosted or cloud
Mattermost workspace. It clusters posts and replies by ``root_id`` into unified,
searchable thread documents so AI agents understand the complete context of team
discussions, technical decisions, and incident channels.

Quickstart::

    import cognee
    from cognee.tasks.ingestion.connectors import mattermost_source

    await cognee.add(
        mattermost_source(
            mattermost_url="https://mattermost.internal.company.com",
            access_token="pat_your_personal_access_token",
            channels=["incidents", "engineering-architecture"],
        )
    )
    await cognee.cognify()

Design & Technical Principles
-----------------------------
* **Thread Clustering**: Mattermost posts in a thread share a ``root_id``. A lone
  reply ("confirmed, shipping now") has zero semantic value without its parent.
  This connector reconstructs threads by grouping replies under their root post
  into a single, structured markdown document.
* **Deterministic Output**: Author headers, ISO timestamps, and reply ordering
  are rendered deterministically so repeated syncs do not produce duplicate
  graph nodes or trigger redundant re-embeddings.
* **Incremental Synchronization**: Uses Mattermost's ``/channels/{id}/posts?since=``
  cursor on ``update_at``. Because editing a reply or adding a comment updates the
  root post's ``update_at``, updated threads are automatically re-rendered.
* **Safe Millisecond Overlap**: To prevent missing posts written during the same
  millisecond as the cursor boundary, queries use ``since = cursor - 1`` and
  deduplicate previously seen post IDs.
* **Filter System & Audit Noise**:
  - System messages (joins, leaves, header changes) have non-empty ``type`` and
    are filtered out.
  - Mattermost creates audit clones when a post is edited (a separate post with
    ``original_id`` set and ``delete_at`` populated). Treating this as a deletion
    would cause phantom deletes; posts with ``original_id`` are cleanly skipped.
* **Forget-on-Delete Lifecycle**: When a post is deleted in Mattermost, the ``since``
  endpoint returns it with ``delete_at > 0``. The connector emits a ``_deleted=True``
  tombstone so Cognee's orphan cleanup prunes it from the knowledge graph.
* **Security & Token Masking**: Personal Access Tokens and Bot Tokens are masked
  in ``__repr__`` and custom exceptions, preventing credentials from leaking into
  logs or telemetry.
* **Scoping**: Documents are scoped under ``cognee_node_set = ["mattermost:<host>/<channel>"]``.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 200
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class MattermostSettings(BaseSettings):
    """Configuration settings for Mattermost ingestion."""

    mattermost_url: str = Field(
        default="http://localhost:8065",
        validation_alias="MATTERMOST_URL",
        description="Base URL of the Mattermost server (e.g. https://chat.company.com).",
    )
    mattermost_token: str = Field(
        default="",
        validation_alias="MATTERMOST_BOT_TOKEN",
        description="Personal Access Token or Bot Token for authentication.",
    )
    mattermost_team: str | None = Field(
        default=None,
        validation_alias="MATTERMOST_TEAM",
        description="Optional team name or ID to restrict channel discovery.",
    )
    mattermost_channels: list[str] = Field(
        default_factory=list,
        validation_alias="MATTERMOST_CHANNELS",
        description="Optional list of channel names or IDs to ingest.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class MattermostSourceError(RuntimeError):
    """Base error for Mattermost connector failures."""


class MattermostAPIError(MattermostSourceError):
    """Mattermost REST API returned an error or could not be reached."""


class MattermostAuthError(MattermostSourceError):
    """Authentication failed (HTTP 401 or 403)."""


class MattermostRateLimitedError(MattermostSourceError):
    """Mattermost returned HTTP 429 Too Many Requests."""

    def __init__(self, retry_after: int | None = None):
        super().__init__("Mattermost API rate limit exceeded")
        self.retry_after = retry_after


class MattermostResourceNotFoundError(MattermostAPIError):
    """Requested channel or team was not found or is inaccessible."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class MattermostClient:
    """Synchronous REST API v4 client for Mattermost with token masking."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        if not base_url or not base_url.strip():
            raise ValueError("Mattermost base_url must not be empty")
        if not token or not token.strip():
            raise ValueError("Mattermost token must not be empty")

        clean_url = base_url.strip().rstrip("/")
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"https://{clean_url}"

        self.base_url = clean_url
        self._token = token.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self._user_cache: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"MattermostClient(base_url={self.base_url!r}, token='***', timeout={self.timeout})"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cognee-mattermost-connector/1.0",
        }

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/api/v4/{endpoint.lstrip('/')}"
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                url = f"{url}?{query}"

        req = urllib.request.Request(url, headers=self._headers(), method="GET")

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if not raw:
                        return None
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise MattermostAuthError(
                        f"Mattermost authentication failed (HTTP {exc.code})"
                    ) from None
                if exc.code == 404:
                    raise MattermostResourceNotFoundError(
                        f"Mattermost resource not found (HTTP 404): {endpoint}"
                    ) from None
                if exc.code == 429:
                    retry_after = None
                    try:
                        retry_after = int(exc.headers.get("Retry-After", 1))
                    except (TypeError, ValueError):
                        pass
                    if attempt < self.max_retries:
                        continue
                    raise MattermostRateLimitedError(retry_after=retry_after) from None
                if attempt == self.max_retries:
                    raise MattermostAPIError(
                        f"Mattermost API error (HTTP {exc.code}) on {endpoint}"
                    ) from None
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise MattermostAPIError(
                        f"Network failure connecting to Mattermost: {exc.reason}"
                    ) from None

        raise MattermostAPIError(f"Exceeded retries requesting {endpoint}")

    def get_me(self) -> dict[str, Any]:
        """Fetch current authenticated user profile."""
        data = self._request("users/me")
        if isinstance(data, dict):
            username = data.get("username", "unknown")
            user_id = data.get("id")
            if user_id:
                self._user_cache[user_id] = username
            return data
        return {}

    def get_username(self, user_id: str) -> str:
        """Resolve user_id to username with local in-memory caching."""
        if not user_id:
            return "system"
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        try:
            user_data = self._request(f"users/{user_id}")
            if isinstance(user_data, dict):
                username = user_data.get("username") or user_data.get("id", user_id)
                self._user_cache[user_id] = username
                return username
        except MattermostSourceError:
            logger.debug("Could not resolve username for user_id: %s", user_id)
        return user_id

    def list_my_channels(self) -> list[dict[str, Any]]:
        """List all channels accessible to the authenticated user."""
        data = self._request("users/me/channels")
        return data if isinstance(data, list) else []

    def get_channel(self, channel_id: str) -> dict[str, Any]:
        """Get channel details including name, display name, and delete status."""
        data = self._request(f"channels/{channel_id}", params={"include_deleted": "true"})
        return data if isinstance(data, dict) else {}

    def get_channel_by_name(self, team_name: str, channel_name: str) -> dict[str, Any]:
        """Get a channel by team name and channel name."""
        data = self._request(f"teams/name/{team_name}/channels/name/{channel_name}")
        return data if isinstance(data, dict) else {}

    def get_channel_posts(
        self,
        channel_id: str,
        since: int | None = None,
        page: int = 0,
        per_page: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Fetch posts for a channel, optionally filtered by `since` timestamp in ms."""
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if since is not None and since > 0:
            params["since"] = since

        data = self._request(f"channels/{channel_id}/posts", params=params)
        return data if isinstance(data, dict) else {"order": [], "posts": {}}

    def get_post_thread(self, post_id: str) -> dict[str, Any]:
        """Fetch complete thread for a post."""
        data = self._request(f"posts/{post_id}/thread")
        return data if isinstance(data, dict) else {"order": [], "posts": {}}


# ---------------------------------------------------------------------------
# Formatting & Thread Clustering
# ---------------------------------------------------------------------------
def _format_timestamp_ms(ts_ms: int | None) -> str:
    if not ts_ms or ts_ms <= 0:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OSError):
        return "N/A"


def _render_thread_markdown(
    root_post: dict[str, Any],
    replies: list[dict[str, Any]],
    channel_info: dict[str, Any],
    server_host: str,
    resolve_username: Callable[[str], str],
) -> str:
    """Render a root post and its replies into a clean, deterministic markdown document."""
    channel_name = channel_info.get("name", "general")
    channel_display = channel_info.get("display_name", channel_name)
    root_id = root_post.get("id", "unknown")
    root_author = resolve_username(root_post.get("user_id", ""))
    created_str = _format_timestamp_ms(root_post.get("create_at"))
    updated_str = _format_timestamp_ms(root_post.get("update_at"))

    # Sort replies chronologically for determinism
    sorted_replies = sorted(replies, key=lambda r: (r.get("create_at") or 0, r.get("id") or ""))

    lines = [
        f"# Mattermost: #{channel_name} - Thread {root_id}",
        "",
        "## Context & Channel Metadata",
        f"- **Channel**: `#{channel_name}` ({channel_display})",
        f"- **Server**: `{server_host}`",
        f"- **Thread Root ID**: `{root_id}`",
        f"- **Author**: @{root_author}",
        f"- **Created**: {created_str}",
        f"- **Last Updated**: {updated_str}",
        f"- **Total Replies**: {len(sorted_replies)}",
        "",
        "## Root Message",
        root_post.get("message", "").strip(),
        "",
    ]

    # Props / Attachments on root if present
    props = root_post.get("props") or {}
    attachments = props.get("attachments") or []
    if attachments and isinstance(attachments, list):
        lines.append("### Attachments & Cards")
        for att in attachments:
            if isinstance(att, dict):
                title = att.get("title") or att.get("fallback") or "Attachment"
                text = att.get("text") or ""
                lines.append(f"- **{title}**: {text}".strip())
        lines.append("")

    if sorted_replies:
        lines.append(f"## Thread Discussion ({len(sorted_replies)} replies)")
        for reply in sorted_replies:
            reply_author = resolve_username(reply.get("user_id", ""))
            reply_time = _format_timestamp_ms(reply.get("create_at"))
            msg = reply.get("message", "").strip()
            lines.append(f"### @{reply_author} *({reply_time})*")
            lines.append(msg if msg else "*(Empty message)*")
            lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Stream Iterator
# ---------------------------------------------------------------------------
def _is_system_post(post: dict[str, Any]) -> bool:
    """Return True if the post is a system join/leave/header message."""
    post_type = post.get("type")
    return bool(post_type and post_type.strip())


def _is_edit_audit_clone(post: dict[str, Any]) -> bool:
    """Return True if the post is an edit-history copy rather than an actual post."""
    original_id = post.get("original_id")
    return bool(original_id and original_id.strip())


def _iter_channel_rows(
    client: MattermostClient,
    channel: dict[str, Any],
    channel_state: dict[str, Any],
    stats: dict[str, int],
    server_host: str,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield document rows and tombstone records for a single channel."""
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)

    # State: last seen cursor (update_at in ms) and seen IDs at that cursor
    cursor = channel_state.get("cursor", 0)
    seen_ids = set(channel_state.get("seen_ids", []))

    # Query with 1 ms overlap to avoid race conditions on identical update_at
    query_since = max(0, cursor - 1) if cursor > 0 else 0

    page = 0
    max_update_at_seen = cursor
    current_seen_at_max = set(seen_ids)

    # Temporary collection of active posts and deletions during this sync run
    raw_posts_by_id: dict[str, dict[str, Any]] = {}
    threads_to_rebuild: set[str] = set()

    while True:
        if check_active is not None:
            check_active()

        try:
            res = client.get_channel_posts(
                channel_id=channel_id,
                since=query_since if query_since > 0 else None,
                page=page,
                per_page=DEFAULT_PAGE_SIZE,
            )
        except MattermostRateLimitedError:
            logger.warning("Mattermost rate limit hit. Halting channel sync cleanly.")
            stats["rate_limited"] = stats.get("rate_limited", 0) + 1
            break

        posts_dict = res.get("posts") or {}
        order = res.get("order") or []

        if not order:
            break

        for post_id in order:
            post = posts_dict.get(post_id)
            if not post or not isinstance(post, dict):
                continue

            stats["scanned"] = stats.get("scanned", 0) + 1

            # Skip system posts
            if _is_system_post(post):
                stats["skipped"] = stats.get("skipped", 0) + 1
                continue

            # Skip edit-audit history copies
            if _is_edit_audit_clone(post):
                stats["skipped"] = stats.get("skipped", 0) + 1
                continue

            update_at = post.get("update_at") or post.get("create_at") or 0
            if update_at < cursor:
                continue
            if update_at == cursor and post_id in seen_ids:
                continue

            # Check deletion
            delete_at = post.get("delete_at") or 0
            if delete_at > 0:
                # Deleted post or thread
                root_id = post.get("root_id") or post_id
                if root_id == post_id:
                    # Root post deleted: emit tombstone for the whole thread
                    yield {
                        "id": f"mattermost:thread:{post_id}",
                        "_deleted": True,
                    }
                    stats["deleted"] = stats.get("deleted", 0) + 1
                else:
                    # Reply deleted: mark thread for re-render
                    threads_to_rebuild.add(root_id)
                continue

            raw_posts_by_id[post_id] = post
            root_id = post.get("root_id") or post_id
            threads_to_rebuild.add(root_id)

            if update_at > max_update_at_seen:
                max_update_at_seen = update_at
                current_seen_at_max = {post_id}
            elif update_at == max_update_at_seen:
                current_seen_at_max.add(post_id)

        # In since queries, Mattermost returns all modified posts. If order < page_size, we are done.
        if len(order) < DEFAULT_PAGE_SIZE:
            break
        page += 1

    # Now render and emit complete thread documents
    for root_id in sorted(threads_to_rebuild):
        # Fetch or resolve full thread from client to guarantee full conversation context
        try:
            thread_data = client.get_post_thread(root_id)
            thread_posts = thread_data.get("posts") or {}
            root_post = thread_posts.get(root_id)
        except MattermostResourceNotFoundError:
            # Root was hard-deleted upstream
            yield {
                "id": f"mattermost:thread:{root_id}",
                "_deleted": True,
            }
            stats["deleted"] = stats.get("deleted", 0) + 1
            continue
        except MattermostSourceError as exc:
            logger.warning("Could not fetch full thread %s: %s", root_id, exc)
            root_post = raw_posts_by_id.get(root_id)

        if not root_post:
            continue

        if (root_post.get("delete_at") or 0) > 0:
            yield {
                "id": f"mattermost:thread:{root_id}",
                "_deleted": True,
            }
            stats["deleted"] = stats.get("deleted", 0) + 1
            continue

        # Collect replies
        replies: list[dict[str, Any]] = []
        for pid, p in thread_posts.items():
            if (
                pid != root_id
                and not _is_system_post(p)
                and not _is_edit_audit_clone(p)
                and (p.get("delete_at") or 0) == 0
            ):
                replies.append(p)

        rendered_text = _render_thread_markdown(
            root_post=root_post,
            replies=replies,
            channel_info=channel,
            server_host=server_host,
            resolve_username=client.get_username,
        )

        doc_id = f"mattermost:thread:{root_id}"
        web_url = f"{client.base_url}/_redirect/pl/{root_id}"

        yield {
            "id": doc_id,
            "title": f"Mattermost: #{channel_name} - Thread {root_id}",
            "content": rendered_text,
            "url": web_url,
            "_deleted": False,
            "cognee_node_set": [f"mattermost:{server_host}/{channel_name}"],
            "metadata": {
                "source": "mattermost",
                "channel_id": channel_id,
                "channel_name": channel_name,
                "root_id": root_id,
                "author": client.get_username(root_post.get("user_id", "")),
                "reply_count": len(replies),
                "created_at": root_post.get("create_at"),
                "updated_at": root_post.get("update_at"),
            },
        }
        stats["emitted"] = stats.get("emitted", 0) + 1

    # Update state cursor
    channel_state["cursor"] = max_update_at_seen
    channel_state["seen_ids"] = list(current_seen_at_max)


def _iter_all_rows(
    client: MattermostClient,
    selected_channels: list[str] | None,
    state: dict[str, Any],
    stats: dict[str, int],
    server_host: str,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate through all targeted channels and yield document streams."""
    channel_states = state.setdefault("channels", {})

    # Discover accessible channels
    accessible = client.list_my_channels()
    accessible_by_id = {c["id"]: c for c in accessible if "id" in c}
    accessible_by_name = {c.get("name"): c for c in accessible if "name" in c}

    target_channels: list[dict[str, Any]] = []

    if selected_channels:
        for ch_key in selected_channels:
            if ch_key in accessible_by_id:
                target_channels.append(accessible_by_id[ch_key])
            elif ch_key in accessible_by_name:
                target_channels.append(accessible_by_name[ch_key])
            else:
                # Direct lookup
                try:
                    ch_info = client.get_channel(ch_key)
                    if ch_info and "id" in ch_info:
                        target_channels.append(ch_info)
                except MattermostSourceError:
                    logger.warning("Could not find channel '%s'", ch_key)
    else:
        target_channels = list(accessible_by_id.values())

    for channel in target_channels:
        ch_id = channel["id"]
        ch_state = channel_states.setdefault(ch_id, {})
        yield from _iter_channel_rows(
            client=client,
            channel=channel,
            channel_state=ch_state,
            stats=stats,
            server_host=server_host,
            check_active=check_active,
        )


# ---------------------------------------------------------------------------
# Public DLT Source Factory
# ---------------------------------------------------------------------------
def mattermost_source(
    *,
    mattermost_url: str | None = None,
    access_token: str | None = None,
    channels: list[str] | None = None,
    resource_name: str | None = None,
    client: MattermostClient | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a dlt resource that yields Mattermost conversation threads for ``remember``.

    Args:
        mattermost_url: Base URL of the Mattermost server. Defaults to ``MATTERMOST_URL`` from .env.
        access_token: Personal Access Token or Bot Token. Defaults to ``MATTERMOST_BOT_TOKEN`` from .env.
        channels: Optional list of channel names or IDs to ingest. If omitted, ingests all accessible channels.
        resource_name: Stable resource table name (defaults to ``mattermost_threads``).
        client: Optional pre-constructed ``MattermostClient`` for testing or custom transport.
        check_active: Optional health/authorization callback called before network requests.

    Returns:
        A dlt resource configured with ``primary_key="id"``, ``write_disposition="merge"``,
        and ``_deleted`` tombstone tracking.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Mattermost connector requires dlt. Install cognee dependencies."
        ) from exc

    if client is None:
        settings = MattermostSettings()
        url = mattermost_url or settings.mattermost_url
        token = access_token or settings.mattermost_token
        target_channels = channels or settings.mattermost_channels or None
        client = MattermostClient(base_url=url, token=token)
    else:
        target_channels = channels

    server_host = urlparse(client.base_url).netloc or "mattermost"
    table_name = resource_name or "mattermost_threads"
    stats: dict[str, int] = {}

    @dlt.resource(
        name=table_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def mattermost_threads():
        stats.clear()
        stats.update(scanned=0, skipped=0, emitted=0, deleted=0)
        state = dlt.current.resource_state()
        yield from _iter_all_rows(
            client=client,
            selected_channels=target_channels,
            state=state,
            stats=stats,
            server_host=server_host,
            check_active=check_active,
        )

    resource = mattermost_threads()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "mattermost")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, table_name)
    resource.cognee_sync_stats = stats
    return resource
