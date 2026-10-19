"""Fathom meeting notes data-source connector for cognee: turn calls into memory.

One source ingests meeting summaries, action items, decisions, and transcripts
from Fathom (https://fathom.video). It structures discussions and follow-ups into
rich markdown documents so AI agents can answer questions like:
"What did we decide about the release timeline in yesterday's call?" or
"Who owns the follow-up task for the database migration?"

Quickstart::

    import cognee
    from cognee.tasks.ingestion.connectors import fathom_source

    await cognee.add(
        fathom_source(
            api_key="your_fathom_api_key",
            include_transcript=True,
        )
    )
    await cognee.cognify()

Design & Technical Principles
-----------------------------
* **Meeting Intelligence & Action Items**: Ingests meeting titles, dates, invitees,
  executive overviews, bulleted key takeaways, and structured action items
  (with assignee names and completion states).
* **Deterministic Formatting**: Meeting metadata, action items, and transcripts
  are rendered in consistent chronological order so repeated syncs produce
  byte-identical documents without duplicate graph entities.
* **Opt-in Transcripts**: Full transcripts can be significantly larger than summaries.
  The connector provides `include_transcript=False` by default for fast, cost-effective
  summary memory, with opt-in support for complete speaker-attributed dialogue.
* **Incremental Sync & Action-Item Re-check**: Uses `created_after` cursor pagination.
  Because action items can be marked complete days after a meeting, the connector
  features a configurable `recheck_days` window (default 7 days) to refresh recent
  meeting states while avoiding full re-downloads.
* **Deletion Reconciliation**: Fathom's API has no dedicated deleted-meeting event
  feed. The connector reconciles previously seen meeting IDs against active listings
  and emits `_deleted=True` tombstones so Cognee's `orphan_cleanup` clears deleted
  calls from the knowledge graph.
* **Token Hygiene & Security**: API keys are masked in `__repr__` and custom exceptions
  so secrets never leak into logs or telemetry.
* **Scoping**: Documents are scoped under ``cognee_node_set = ["fathom:meetings"]``.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.fathom.video/external/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_RECHECK_DAYS = 7


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class FathomSettings(BaseSettings):
    """Configuration settings for Fathom ingestion."""

    fathom_api_key: str = Field(
        default="",
        validation_alias="FATHOM_API_KEY",
        description="Fathom API key from Fathom settings.",
    )
    fathom_base_url: str = Field(
        default=DEFAULT_BASE_URL,
        validation_alias="FATHOM_BASE_URL",
        description="Base URL for Fathom REST API v1.",
    )
    fathom_include_transcript: bool = Field(
        default=False,
        validation_alias="FATHOM_INCLUDE_TRANSCRIPT",
        description="Whether to include full speaker-attributed transcripts.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class FathomSourceError(RuntimeError):
    """Base error for Fathom connector operations."""


class FathomAPIError(FathomSourceError):
    """Fathom REST API returned an error or could not be reached."""


class FathomAuthError(FathomSourceError):
    """Authentication failed (HTTP 401 or 403)."""


class FathomRateLimitedError(FathomSourceError):
    """Fathom returned HTTP 429 Too Many Requests."""

    def __init__(self, retry_after: int | None = None):
        super().__init__("Fathom API rate limit exceeded")
        self.retry_after = retry_after


class FathomResourceNotFoundError(FathomAPIError):
    """Requested meeting was not found (HTTP 404)."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class FathomClient:
    """Synchronous REST API client for Fathom with key masking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("Fathom api_key must not be empty")

        clean_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"https://{clean_url}"

        self.api_key = api_key.strip()
        self.base_url = clean_url
        self.timeout = timeout
        self.max_retries = max_retries

    def __repr__(self) -> str:
        return f"FathomClient(base_url={self.base_url!r}, api_key='***', timeout={self.timeout})"

    def _headers(self) -> dict[str, str]:
        return {
            "X-Api-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cognee-fathom-connector/1.0",
        }

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urllib.parse.urlencode(clean_params)}"

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
                    raise FathomAuthError(
                        f"Fathom authentication failed (HTTP {exc.code})"
                    ) from None
                if exc.code == 404:
                    raise FathomResourceNotFoundError(
                        f"Fathom resource not found (HTTP 404): {endpoint}"
                    ) from None
                if exc.code == 429:
                    retry_after = None
                    try:
                        retry_after = int(exc.headers.get("Retry-After", 1))
                    except (TypeError, ValueError):
                        pass
                    if attempt < self.max_retries:
                        continue
                    raise FathomRateLimitedError(retry_after=retry_after) from None
                if attempt == self.max_retries:
                    raise FathomAPIError(
                        f"Fathom API error (HTTP {exc.code}) on {endpoint}"
                    ) from None
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise FathomAPIError(
                        f"Network failure connecting to Fathom: {exc.reason}"
                    ) from None

        raise FathomAPIError(f"Exceeded retries requesting {endpoint}")

    def list_meetings(
        self,
        created_after: str | None = None,
        created_before: str | None = None,
        cursor: str | None = None,
        include_transcript: bool = False,
    ) -> dict[str, Any]:
        """Fetch meetings list with optional cursor and transcript toggle."""
        params: dict[str, Any] = {}
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        if include_transcript:
            params["include_transcript"] = "true"

        data = self._request("meetings", params=params)
        return data if isinstance(data, dict) else {"items": [], "next_cursor": None}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _format_seconds(seconds: float | None) -> str:
    """Format duration in seconds to MM:SS or HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00"
    total_sec = int(seconds)
    hours, remainder = divmod(total_sec, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _render_meeting_markdown(meeting: dict[str, Any], include_transcript: bool = False) -> str:
    """Render a Fathom meeting object into clean, deterministic markdown."""
    title = (meeting.get("title") or meeting.get("meeting_title") or "Untitled Meeting").strip()
    meeting_id = str(meeting.get("id", "unknown"))
    created_at = meeting.get("created_at") or "N/A"
    recording_start = meeting.get("recording_start_time") or "N/A"
    duration_sec = meeting.get("duration")
    duration_str = _format_seconds(duration_sec) if duration_sec is not None else "N/A"
    share_url = meeting.get("url") or "N/A"

    lines = [
        f"# Meeting: {title}",
        "",
        "## Overview & Metadata",
        f"- **Meeting ID**: `{meeting_id}`",
        f"- **Recorded Date**: {recording_start}",
        f"- **Created At**: {created_at}",
        f"- **Duration**: {duration_str}",
        f"- **Recording URL**: {share_url}",
    ]

    # Invitees
    invitees = meeting.get("invitees") or []
    if invitees:
        attendee_names = []
        for inv in sorted(invitees, key=lambda x: str(x.get("email") or x.get("name") or "")):
            name = inv.get("name") or inv.get("email")
            if name:
                attendee_names.append(str(name).strip())
        if attendee_names:
            lines.append(f"- **Attendees**: {', '.join(attendee_names)}")

    lines.append("")

    # Default Summary
    summary = meeting.get("default_summary") or {}
    overview = summary.get("overview")
    key_points = summary.get("key_points") or []

    if overview or key_points:
        lines.append("## Executive Summary")
        if overview:
            lines.append(str(overview).strip())
            lines.append("")
        if key_points:
            lines.append("### Key Points & Decisions")
            for pt in key_points:
                lines.append(f"- {str(pt).strip()}")
            lines.append("")

    # Action Items
    action_items = meeting.get("action_items") or []
    if action_items:
        lines.append("## Action Items & Next Steps")
        for item in sorted(action_items, key=lambda x: str(x.get("description") or "")):
            desc = str(item.get("description") or "").strip()
            completed = bool(item.get("completed"))
            checkbox = "[x]" if completed else "[ ]"
            assignee = item.get("assignee") or {}
            assignee_name = assignee.get("name") or assignee.get("email") or "Unassigned"
            status_text = "Completed" if completed else "Pending"
            lines.append(
                f"- {checkbox} **{desc}** (Assignee: @{assignee_name}, Status: {status_text})"
            )
        lines.append("")

    # Transcript
    if include_transcript:
        transcript = meeting.get("transcript") or []
        if transcript:
            lines.append("## Dialogue & Discussion Transcript")
            for entry in transcript:
                speaker = entry.get("speaker") or {}
                speaker_name = speaker.get("name") or "Speaker"
                start_time = _format_seconds(entry.get("start_time"))
                text = str(entry.get("text") or "").strip()
                if text:
                    lines.append(f"- **{speaker_name}** *({start_time})*: {text}")
            lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Stream Iterator
# ---------------------------------------------------------------------------
def _iter_fathom_rows(
    client: FathomClient,
    include_transcript: bool,
    recheck_days: int,
    state: dict[str, Any],
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate through Fathom meetings, emit documents and deletion tombstones."""
    cursor_timestamp: str | None = state.get("cursor")
    seen_ids: set[str] = set(state.get("seen_ids", []))

    # Calculate created_after using recheck window for action item updates
    query_after: str | None = None
    if cursor_timestamp:
        try:
            dt = datetime.fromisoformat(cursor_timestamp.replace("Z", "+00:00"))
            recheck_dt = dt - timedelta(days=recheck_days)
            query_after = recheck_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            query_after = cursor_timestamp

    max_created_seen = cursor_timestamp
    active_meeting_ids: set[str] = set()
    cursor_token: str | None = None

    while True:
        if check_active is not None:
            check_active()

        try:
            res = client.list_meetings(
                created_after=query_after,
                cursor=cursor_token,
                include_transcript=include_transcript,
            )
        except FathomRateLimitedError:
            logger.warning("Fathom rate limit reached. Halting sync cleanly.")
            stats["rate_limited"] = stats.get("rate_limited", 0) + 1
            break

        items = res.get("items") or []
        if not items:
            break

        for meeting in items:
            meeting_id = str(meeting.get("id"))
            if not meeting_id or meeting_id == "None":
                continue

            active_meeting_ids.add(meeting_id)
            stats["scanned"] = stats.get("scanned", 0) + 1

            created_at = meeting.get("created_at") or ""
            if created_at and (not max_created_seen or created_at > max_created_seen):
                max_created_seen = created_at

            rendered_content = _render_meeting_markdown(
                meeting=meeting,
                include_transcript=include_transcript,
            )

            title = (meeting.get("title") or meeting.get("meeting_title") or "Meeting").strip()
            share_url = meeting.get("url") or f"https://fathom.video/share/{meeting_id}"

            yield {
                "id": f"fathom:meeting:{meeting_id}",
                "title": f"Meeting: {title}",
                "content": rendered_content,
                "url": share_url,
                "_deleted": False,
                "cognee_node_set": ["fathom:meetings"],
                "metadata": {
                    "source": "fathom",
                    "meeting_id": meeting_id,
                    "created_at": created_at,
                    "recording_start_time": meeting.get("recording_start_time"),
                    "duration": meeting.get("duration"),
                    "has_transcript": include_transcript,
                },
            }
            stats["emitted"] = stats.get("emitted", 0) + 1

        cursor_token = res.get("next_cursor")
        if not cursor_token:
            break

    # Deletion reconciliation:
    # If this was a full or recheck sync, check if previously seen meetings disappeared
    if seen_ids and not query_after:
        # Full sync: any previously seen meeting missing from active_meeting_ids is deleted
        deleted_ids = seen_ids - active_meeting_ids
        for did in deleted_ids:
            yield {
                "id": f"fathom:meeting:{did}",
                "_deleted": True,
            }
            stats["deleted"] = stats.get("deleted", 0) + 1

    # Update state
    if max_created_seen:
        state["cursor"] = max_created_seen
    state["seen_ids"] = list(seen_ids | active_meeting_ids)


# ---------------------------------------------------------------------------
# Public DLT Source Factory
# ---------------------------------------------------------------------------
def fathom_source(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    include_transcript: bool | None = None,
    recheck_days: int = DEFAULT_RECHECK_DAYS,
    resource_name: str | None = None,
    client: FathomClient | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a dlt resource that yields Fathom meeting notes and action items for ``remember``.

    Args:
        api_key: Fathom API Key from Fathom settings. Defaults to ``FATHOM_API_KEY`` from .env.
        base_url: Base URL of the Fathom REST API. Defaults to ``https://api.fathom.video/external/v1``.
        include_transcript: Whether to include complete speaker-attributed transcripts (default False).
        recheck_days: Rolling window in days to recheck recent meetings for action-item updates (default 7).
        resource_name: Stable resource table name (defaults to ``fathom_meetings``).
        client: Optional pre-constructed ``FathomClient`` for custom transport or testing.
        check_active: Optional health/authorization callback called before network requests.

    Returns:
        A dlt resource configured with ``primary_key="id"``, ``write_disposition="merge"``,
        and ``_deleted`` tombstone tracking.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Fathom connector requires dlt. Install cognee dependencies."
        ) from exc

    if client is None:
        settings = FathomSettings()
        key = api_key or settings.fathom_api_key
        url = base_url or settings.fathom_base_url
        transcripts = (
            include_transcript
            if include_transcript is not None
            else settings.fathom_include_transcript
        )
        client = FathomClient(api_key=key, base_url=url)
    else:
        transcripts = bool(include_transcript)

    table_name = resource_name or "fathom_meetings"
    stats: dict[str, int] = {}

    @dlt.resource(
        name=table_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def fathom_meetings():
        stats.clear()
        stats.update(scanned=0, emitted=0, deleted=0)
        state = dlt.current.resource_state()
        yield from _iter_fathom_rows(
            client=client,
            include_transcript=transcripts,
            recheck_days=recheck_days,
            state=state,
            stats=stats,
            check_active=check_active,
        )

    resource = fathom_meetings()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "fathom")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, table_name)
    resource.cognee_sync_stats = stats
    return resource
