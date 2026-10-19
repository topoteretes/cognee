"""Sentry connector for Cognee.

Ingests runtime error telemetry, exception groups, culprit stack traces, and incident
metadata from Sentry into Cognee's memory layer using DLT document sources.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator
from typing import Any

import dlt  # type: ignore

from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    NODE_SET_COLUMN_HINT,
    PIPELINE_SCOPE_ATTR,
    guarded_rows,
)

logger = logging.getLogger(__name__)

DEFAULT_SENTRY_BASE_URL = "https://sentry.io"
_RATE_LIMIT_FLOOR = 5


class SentryError(Exception):
    """Base error for Sentry API connector failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SentryAuthenticationError(SentryError):
    """Raised on invalid authentication tokens (HTTP 401)."""


class SentryPermissionError(SentryError):
    """Raised when organization or project access is forbidden (HTTP 403)."""


class SentryNotFoundError(SentryError):
    """Raised when the organization or project does not exist (HTTP 404)."""


class SentryRateLimitError(SentryError):
    """Raised when rate limits are exceeded (HTTP 429)."""


def _mask_token(token: str | None) -> str:
    """Safely mask token for logs and representation."""
    if not token:
        return "<none>"
    if len(token) <= 8:
        return "sentry_***"
    return f"{token[:4]}...{token[-4:]}"


def _parse_link_header(link_header: str | None) -> tuple[str | None, bool]:
    """Parse Sentry RFC-5988 Link header for rel='next' cursor and has_more indicator.

    Sentry headers look like:
    <https://sentry.io/api/0/projects/org/proj/issues/?cursor=100:0:0>; rel="next"; results="true"
    """
    if not link_header:
        return None, False

    next_cursor = None
    has_results = False

    # Split links by comma
    parts = link_header.split(",")
    for part in parts:
        segments = [s.strip() for s in part.split(";")]
        url_match = re.search(r"<([^>]+)>", segments[0])
        if not url_match:
            continue
        url = url_match.group(1)

        rel_is_next = False
        results_true = False

        for param in segments[1:]:
            if 'rel="next"' in param or "rel='next'" in param or "rel=next" in param:
                rel_is_next = True
            if 'results="true"' in param or "results='true'" in param or "results=true" in param:
                results_true = True

        if rel_is_next:
            has_results = results_true
            parsed_url = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            if query_params.get("cursor"):
                next_cursor = query_params["cursor"][0]
            else:
                next_cursor = url

    return next_cursor, has_results


class SentryClient:
    """Synchronous HTTP client for Sentry REST API v0."""

    def __init__(
        self,
        auth_token: str,
        base_url: str = DEFAULT_SENTRY_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not auth_token or not auth_token.strip():
            raise SentryAuthenticationError("Sentry auth_token cannot be empty.")

        self._token = auth_token.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"<SentryClient base_url='{self.base_url}' token='{_mask_token(self._token)}'>"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "User-Agent": "cognee-sentry-connector/1.0",
        }

    def request(
        self,
        endpoint_path: str,
        query_params: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, str]]:
        """Perform HTTP request against Sentry API and return (json_data, headers)."""
        url = f"{self.base_url}/{endpoint_path.lstrip('/')}"
        if query_params:
            encoded_params = urllib.parse.urlencode(
                {k: v for k, v in query_params.items() if v is not None}
            )
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{encoded_params}"

        req = urllib.request.Request(url, headers=self._build_headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                headers = {k.lower(): v for k, v in response.headers.items()}
                raw_body = response.read().decode("utf-8")
                try:
                    data = json.loads(raw_body)
                except json.JSONDecodeError as err:
                    raise SentryError(f"Failed to parse Sentry JSON response: {err}") from err
                return data, headers
        except urllib.error.HTTPError as http_err:
            status = http_err.code
            if status == 401:
                raise SentryAuthenticationError(
                    "Invalid or expired Sentry authentication token.", status_code=status
                ) from http_err
            if status == 403:
                raise SentryPermissionError(
                    f"Permission denied for Sentry endpoint: {endpoint_path}", status_code=status
                ) from http_err
            if status == 404:
                raise SentryNotFoundError(
                    f"Sentry resource not found: {endpoint_path}", status_code=status
                ) from http_err
            if status == 429:
                retry_after = http_err.headers.get("Retry-After")
                msg = f"Sentry rate limit exceeded (HTTP 429). Retry-After: {retry_after}"
                raise SentryRateLimitError(msg, status_code=status) from http_err
            raise SentryError(
                f"Sentry HTTP request failed with status {status}: {http_err.reason}",
                status_code=status,
            ) from http_err
        except urllib.error.URLError as url_err:
            raise SentryError(f"Network error connecting to Sentry: {url_err.reason}") from url_err

    def fetch_project(self, organization_slug: str, project_slug: str) -> dict[str, Any]:
        """Verify project existence and fetch project metadata."""
        endpoint = f"api/0/projects/{organization_slug}/{project_slug}/"
        data, _ = self.request(endpoint)
        return data

    def list_issues(
        self,
        organization_slug: str,
        project_slug: str,
        query: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        stats_period: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Fetch a page of Sentry issue groups."""
        endpoint = f"api/0/projects/{organization_slug}/{project_slug}/issues/"
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if cursor:
            params["cursor"] = cursor
        if stats_period:
            params["statsPeriod"] = stats_period

        data, headers = self.request(endpoint, query_params=params)
        if not isinstance(data, list):
            raise SentryError(f"Expected list of issues from Sentry, got {type(data).__name__}")
        return data, headers

    def get_latest_event(self, issue_id: str) -> dict[str, Any] | None:
        """Fetch latest event details including stack trace and breadcrumbs."""
        endpoint = f"api/0/issues/{issue_id}/events/latest/"
        try:
            data, _ = self.request(endpoint)
            return data if isinstance(data, dict) else None
        except SentryNotFoundError:
            return None
        except (urllib.error.URLError, SentryError) as exc:
            logger.debug("Failed to fetch latest event for issue %s: %s", issue_id, exc)
            return None


class _SentryWalker:
    """Keyset-aware crawler for Sentry issues and telemetry."""

    def __init__(
        self,
        client: SentryClient,
        organization_slug: str,
        project_slug: str,
        query: str | None = None,
        fetch_events: bool = True,
        max_issues: int | None = None,
    ) -> None:
        self.client = client
        self.organization_slug = organization_slug
        self.project_slug = project_slug
        self.query = query
        self.fetch_events = fetch_events
        self.max_issues = max_issues

    def walk(self, initial_cursor: str | None = None) -> Generator[dict[str, Any], None, None]:
        """Walk issues using cursor pagination with dynamic rate limit protection."""
        cursor = initial_cursor
        total_emitted = 0

        while True:
            try:
                issues, headers = self.client.list_issues(
                    organization_slug=self.organization_slug,
                    project_slug=self.project_slug,
                    query=self.query,
                    cursor=cursor,
                )
            except SentryRateLimitError as rate_err:
                logger.warning("Sentry rate limit reached; ending crawl gracefully: %s", rate_err)
                break
            except Exception as exc:
                logger.error("Error during Sentry issue crawl: %s", exc)
                raise

            # Check rate limit headroom
            remaining = headers.get("x-sentry-rate-limit-remaining")
            if remaining is not None:
                try:
                    if int(remaining) < _RATE_LIMIT_FLOOR:
                        logger.warning(
                            "Sentry remaining rate limit (%s) is below floor (%d); stopping batch.",
                            remaining,
                            _RATE_LIMIT_FLOOR,
                        )
                        for issue in issues:
                            yield self._process_issue(issue)
                        break
                except ValueError:
                    pass

            if not issues:
                break

            for issue in issues:
                yield self._process_issue(issue)
                total_emitted += 1
                if self.max_issues and total_emitted >= self.max_issues:
                    return

            next_cursor, has_more = _parse_link_header(headers.get("link"))
            if not has_more or not next_cursor:
                break

            cursor = next_cursor

    def _process_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Enrich issue with latest event details if requested."""
        issue_id = str(issue.get("id", ""))
        latest_event = None
        if self.fetch_events and issue_id:
            latest_event = self.client.get_latest_event(issue_id)

        return {
            "issue": issue,
            "latest_event": latest_event,
            "organization_slug": self.organization_slug,
            "project_slug": self.project_slug,
        }


def _extract_stacktrace(event: dict[str, Any] | None) -> list[str]:
    """Extract and format stack trace frames from Sentry event entries."""
    if not event:
        return []

    frames_output = []
    entries = event.get("entries", [])
    for entry in entries:
        if entry.get("type") == "exception":
            values = entry.get("data", {}).get("values", [])
            for val in values:
                exc_type = val.get("type", "Error")
                exc_value = val.get("value", "")
                stacktrace = val.get("stacktrace", {})
                frames = stacktrace.get("frames", [])
                for frame in frames:
                    filename = frame.get("filename") or frame.get("absPath") or "unknown"
                    lineno = frame.get("lineNo") or "?"
                    function = frame.get("function") or "unknown"
                    context = frame.get("context", [])
                    line_code = ""
                    if context:
                        for ctx_line in context:
                            if isinstance(ctx_line, (list, tuple)) and len(ctx_line) > 1:
                                line_code = str(ctx_line[1]).strip()
                                break
                    frame_str = f'  File "{filename}", line {lineno}, in {function}'
                    if line_code:
                        frame_str += f"\n    {line_code}"
                    frames_output.append(frame_str)
                if exc_type or exc_value:
                    frames_output.append(f"{exc_type}: {exc_value}")

    return frames_output


def _render_sentry_issue_markdown(
    issue: dict[str, Any],
    latest_event: dict[str, Any] | None,
    org_slug: str,
    project_slug: str,
) -> str:
    """Render a deterministic markdown document for a Sentry error group."""
    issue_id = str(issue.get("id", "unknown"))
    title = issue.get("title") or "Unnamed Sentry Issue"
    culprit = issue.get("culprit") or "unknown"
    status = issue.get("status") or "unresolved"
    level = issue.get("level") or "error"
    count = issue.get("count", "0")
    user_count = issue.get("userCount", "0")
    first_seen = issue.get("firstSeen") or "unknown"
    last_seen = issue.get("lastSeen") or "unknown"
    permalink = issue.get("permalink") or ""

    lines = [
        f"# Sentry Incident: {title}",
        "",
        f"- **Issue ID**: `{issue_id}`",
        f"- **Project**: `{org_slug}/{project_slug}`",
        f"- **Status**: `{status}`",
        f"- **Level**: `{level}`",
        f"- **Culprit**: `{culprit}`",
        f"- **Occurrences**: `{count}`",
        f"- **Users Affected**: `{user_count}`",
        f"- **First Seen**: `{first_seen}`",
        f"- **Last Seen**: `{last_seen}`",
    ]

    if permalink:
        lines.append(f"- **Permalink**: {permalink}")

    # Render metadata tags sorted deterministically
    tags = []
    if latest_event and "tags" in latest_event:
        raw_tags = latest_event.get("tags", [])
        if isinstance(raw_tags, list):
            for t in raw_tags:
                if isinstance(t, dict) and "key" in t and "value" in t:
                    tags.append(f"{t['key']}={t['value']}")
        elif isinstance(raw_tags, dict):
            for k, v in sorted(raw_tags.items()):
                tags.append(f"{k}={v}")

    tags.sort()
    if tags:
        lines.extend(["", "### Environment & Tags", ""])
        for tag in tags:
            lines.append(f"- `{tag}`")

    # Render stack trace frames
    stacktrace_frames = _extract_stacktrace(latest_event)
    if stacktrace_frames:
        lines.extend(["", "### Culprit Stacktrace", "```python"])
        lines.extend(stacktrace_frames)
        lines.append("```")

    return "\n".join(lines).strip()


def build_sentry_service(
    auth_token: str,
    base_url: str = DEFAULT_SENTRY_BASE_URL,
    timeout: float = 30.0,
) -> SentryClient:
    """Build and return an authenticated SentryClient."""
    return SentryClient(auth_token=auth_token, base_url=base_url, timeout=timeout)


def sentry_source(
    organization_slug: str,
    project_slug: str,
    auth_token: str | None = None,
    client: SentryClient | None = None,
    base_url: str = DEFAULT_SENTRY_BASE_URL,
    query: str | None = None,
    fetch_events: bool = True,
    max_issues: int | None = None,
    check_active: Callable[[], None] | None = None,
):
    """DLT source ingesting Sentry issue groups and telemetry into Cognee memory.

    Rows are formatted into structured markdown and tagged with:
      - ``cognee_node_set = ["sentry:<org>/<project>"]``
      - ``cognee_document_source = "sentry"``
    """
    if client is None:
        if not auth_token:
            raise SentryAuthenticationError("Either 'client' or 'auth_token' must be provided.")
        client = build_sentry_service(auth_token=auth_token, base_url=base_url)

    walker = _SentryWalker(
        client=client,
        organization_slug=organization_slug,
        project_slug=project_slug,
        query=query,
        fetch_events=fetch_events,
        max_issues=max_issues,
    )

    node_set_tag = f"sentry:{organization_slug}/{project_slug}"

    @dlt.resource(
        name="sentry_issues",
        write_disposition="merge",
        primary_key="id",
        columns=NODE_SET_COLUMN_HINT,
    )
    def sentry_issues():
        """Emit Sentry issues as normalized documents with cursor advancement."""
        state = dlt.current.resource_state()
        initial_cursor = state.get("cursor")

        rows_gen = walker.walk(initial_cursor=initial_cursor)

        for item in guarded_rows(rows_gen, check_active=check_active):
            issue = item["issue"]
            latest_event = item["latest_event"]
            issue_id = str(issue.get("id"))

            markdown_doc = _render_sentry_issue_markdown(
                issue=issue,
                latest_event=latest_event,
                org_slug=organization_slug,
                project_slug=project_slug,
            )

            # Update latest state cursor
            state["last_seen_issue_id"] = issue_id

            yield {
                "id": f"sentry:{project_slug}:issue:{issue_id}",
                "text": markdown_doc,
                "title": f"Sentry: {issue.get('title', issue_id)}",
                "issue_id": issue_id,
                "project": project_slug,
                "organization": organization_slug,
                "status": issue.get("status"),
                "level": issue.get("level"),
                "count": issue.get("count"),
                "last_seen": issue.get("lastSeen"),
                NODE_SET_COLUMN: [node_set_tag],
            }

    @dlt.source(name="sentry")
    def _source():
        return [sentry_issues]

    src = _source()
    setattr(src, DOCUMENT_SOURCE_ATTR, "sentry")
    setattr(src, PIPELINE_SCOPE_ATTR, f"sentry:{organization_slug}/{project_slug}")
    return src
