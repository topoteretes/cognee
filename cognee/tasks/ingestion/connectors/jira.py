"""Jira connector for cognee: a ``dlt`` source that turns Jira Cloud projects and issues into memory.

One source is one Jira Cloud instance and a selection of project keys. It yields one document
per issue (with its description, metadata, subtasks, and comments folded in) as flat rows, and
is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import jira_source

    await cognee.remember(
        jira_source(
            base_url="https://acme.atlassian.net",
            access_token="<oauth bearer token or email:api_token>",
            project_keys=["ENG", "OPS"],
        ),
        dataset_name="jira",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are prefixed
  by kind (``jira:issue:<id>``). Every column feeds the stored document identity, so the
  rendering is frozen and deterministic: no volatile timestamps in ``content``, and ``updated``
  lives only in resource state.
* **Node set** ``jira:<project_key>`` per issue row so a ``recall`` can be scoped to a single
  Jira project.
* **Incremental** by ``updated`` timestamp per project key stored in ``dlt.current.resource_state()``,
  plus a ``known_ids`` set per project to detect and tombstone deleted or moved issues on full
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

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
DEFAULT_MAX_REQUESTS = 500
_TIMEOUT = (10, 30)
_RETRIES = 3
_PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")


class JiraSourceError(RuntimeError):
    """Base error raised by the Jira connector."""


class JiraAuthError(JiraSourceError):
    """Jira rejected the credentials (HTTP 401 or 403 on authentication)."""


class JiraRateLimitedError(JiraSourceError):
    """Jira answered HTTP 429 Too Many Requests."""

    def __init__(self, retry_after: int | None = None):
        super().__init__("Jira rate limit exceeded (HTTP 429)")
        self.retry_after = retry_after


class JiraAPIError(JiraSourceError):
    """Jira answered with an unexpected HTTP or payload error."""


def _normalize_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("base_url must be a valid http(s) Jira base URL.")
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _normalize_project_keys(project_keys: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(project_keys, (list, tuple)) or not project_keys:
        raise ValueError("project_keys must be a non-empty list or tuple of Jira project keys.")
    cleaned: set[str] = set()
    for key in project_keys:
        if not isinstance(key, str):
            raise ValueError("Every project key must be a string.")
        norm = key.strip().upper()
        if not _PROJECT_KEY_RE.match(norm):
            raise ValueError(f"Invalid Jira project key: {key!r}")
        cleaned.add(norm)
    return sorted(cleaned)


def _adf_to_text(node: Any) -> str:
    """Convert Atlassian Document Format (ADF) JSON or plain text into deterministic text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type", "")
    if node_type == "text":
        return str(node.get("text") or "")
    if node_type == "hardBreak":
        return "\n"

    children = node.get("content")
    if not isinstance(children, list):
        return ""

    parts = [_adf_to_text(child) for child in children]
    if node_type in ("paragraph", "heading", "blockquote", "codeBlock", "listItem"):
        joined = "".join(parts).strip()
        return f"{joined}\n" if joined else ""
    if node_type in ("bulletList", "orderedList"):
        lines = [p.strip() for p in parts if p.strip()]
        return "\n".join(f"- {line}" for line in lines) + ("\n" if lines else "")
    return "".join(parts).strip()


class JiraClient:
    """Minimal synchronous REST client for Atlassian Jira Cloud v3."""

    def __init__(
        self,
        base_url: str,
        access_token: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = _normalize_base_url(base_url)
        token = (access_token or "").strip()
        if not token or not token.isascii() or not token.isprintable():
            raise ValueError("The Jira access token is empty or has an invalid format.")
        self._access_token = token
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"JiraClient(base_url={self.base_url!r}, access_token=<redacted>)"

    def _headers(self) -> dict[str, str]:
        token = self._access_token
        if ":" in token and " " not in token:
            encoded = base64.b64encode(token.encode("ascii")).decode("ascii")
            auth_header = f"Basic {encoded}"
        else:
            auth_header = f"Bearer {token}"
        return {
            "Authorization": auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import requests

        url = f"{self.base_url}/{path.lstrip('/')}"
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
                    raise JiraAPIError(f"Jira request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue

            status = response.status_code
            if status >= 500 and not last:
                self._sleep(2**attempt)
                continue
            if status in (401, 403):
                raise JiraAuthError(f"Jira rejected credentials (HTTP {status})")
            if status == 429:
                retry_hdr = response.headers.get("Retry-After")
                retry_sec = int(retry_hdr) if retry_hdr and retry_hdr.isdigit() else None
                raise JiraRateLimitedError(retry_after=retry_sec)
            if status == 404:
                raise JiraAPIError("Jira endpoint or entity not found (HTTP 404)")
            if status >= 400:
                raise JiraAPIError(f"Jira request failed with HTTP {status}")

            try:
                data = response.json()
            except ValueError as exc:
                raise JiraAPIError("Jira response is not valid JSON") from exc
            if not isinstance(data, dict):
                raise JiraAPIError("Jira response root must be a JSON object")
            return data

        raise JiraAPIError("Jira request failed")  # pragma: no cover

    def search_issues(
        self,
        jql: str,
        *,
        start_at: int = 0,
        max_results: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        fields = [
            "summary",
            "description",
            "status",
            "priority",
            "issuetype",
            "assignee",
            "reporter",
            "labels",
            "parent",
            "subtasks",
            "comment",
            "updated",
            "project",
        ]
        return self.request_json(
            "GET",
            "/rest/api/3/search",
            params={
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results,
                "fields": ",".join(fields),
            },
        )


def build_jira_service(base_url: str, access_token: str) -> JiraClient:
    """Construct a :class:`JiraClient` for :func:`jira_source`."""
    return JiraClient(base_url=base_url, access_token=access_token)


def _render_issue_row(base_url: str, project_key: str, issue: dict[str, Any]) -> dict[str, Any]:
    issue_id = str(issue.get("id") or "").strip()
    issue_key = str(issue.get("key") or issue_id).strip()
    fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}

    summary = str(fields.get("summary") or issue_key).strip()
    status_name = str((fields.get("status") or {}).get("name") or "Unknown").strip()
    priority_name = str((fields.get("priority") or {}).get("name") or "Unspecified").strip()
    issue_type = str((fields.get("issuetype") or {}).get("name") or "Issue").strip()
    assignee_name = str((fields.get("assignee") or {}).get("displayName") or "Unassigned").strip()
    reporter_name = str((fields.get("reporter") or {}).get("displayName") or "Unknown").strip()

    raw_labels = fields.get("labels")
    labels = sorted(
        {str(lbl).strip() for lbl in (raw_labels if isinstance(raw_labels, list) else []) if str(lbl).strip()}
    )

    parent_key = str((fields.get("parent") or {}).get("key") or "").strip()
    description_text = _adf_to_text(fields.get("description"))

    lines = [
        f"[{issue_key}] {summary}",
        f"Project: {project_key}",
        f"Type: {issue_type}",
        f"Status: {status_name}",
        f"Priority: {priority_name}",
        f"Assignee: {assignee_name}",
        f"Reporter: {reporter_name}",
    ]
    if parent_key:
        lines.append(f"Parent: {parent_key}")
    if labels:
        lines.append(f"Labels: {', '.join(labels)}")

    if description_text:
        lines.append("")
        lines.append("Description:")
        lines.append(description_text)

    subtasks = fields.get("subtasks")
    if isinstance(subtasks, list) and subtasks:
        sub_lines: list[str] = []
        for sub in subtasks:
            if not isinstance(sub, dict):
                continue
            s_key = str(sub.get("key") or "").strip()
            s_fields = sub.get("fields") if isinstance(sub.get("fields"), dict) else {}
            s_sum = str(s_fields.get("summary") or "").strip()
            s_status = str((s_fields.get("status") or {}).get("name") or "").strip()
            if s_key:
                sub_lines.append(f"- {s_key}: {s_sum} ({s_status})" if s_status else f"- {s_key}: {s_sum}")
        if sub_lines:
            lines.append("")
            lines.append("Subtasks:")
            lines.extend(sorted(sub_lines))

    comment_block = fields.get("comment")
    comments = comment_block.get("comments") if isinstance(comment_block, dict) else []
    if isinstance(comments, list) and comments:
        c_lines: list[str] = []
        for c in comments:
            if not isinstance(c, dict):
                continue
            author = str((c.get("author") or {}).get("displayName") or "User").strip()
            body_txt = _adf_to_text(c.get("body"))
            if body_txt:
                c_lines.append(f"- {author}: {body_txt}")
        if c_lines:
            lines.append("")
            lines.append("Comments:")
            lines.extend(c_lines)

    return {
        "id": f"jira:issue:{issue_id}",
        "title": f"{issue_key}: {summary}",
        "content": "\n".join(lines).strip(),
        "url": f"{base_url}/browse/{issue_key}",
        "_deleted": False,
        NODE_SET_COLUMN: [f"jira:{project_key}"],
    }


def _iter_rows(
    client: JiraClient,
    project_keys: list[str],
    state: dict[str, Any],
    stats: dict[str, int],
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    projects_state = state.setdefault("projects", {})
    requests_used = 0

    # Tombstone issues belonging to projects that were deselected since the previous sync
    deselected = sorted(set(projects_state.keys()) - set(project_keys))
    for old_proj in deselected:
        old_entry = projects_state.pop(old_proj, {})
        for issue_id in sorted(old_entry.get("known_ids", [])):
            stats["deleted"] += 1
            yield {
                "id": f"jira:issue:{issue_id}",
                "title": "",
                "content": "",
                "url": "",
                "_deleted": True,
            }

    for project_key in project_keys:
        p_state = projects_state.setdefault(project_key, {"cursor": None, "known_ids": []})
        cursor = p_state.get("cursor")
        known_ids = set(p_state.get("known_ids", []))
        current_run_ids: set[str] = set()
        newest_updated: str | None = cursor

        # When cursor is None (first full sync) or full reconciliation pass, query all issues in project
        jql = f'project = "{project_key}" ORDER BY updated ASC'
        if cursor:
            jql = f'project = "{project_key}" AND updated >= "{cursor}" ORDER BY updated ASC'

        start_at = 0
        while True:
            if requests_used >= max_requests:
                stats["failed"] += 1
                stats["failed_budget"] += 1
                return

            try:
                requests_used += 1
                payload = client.search_issues(jql, start_at=start_at, max_results=page_size)
            except JiraRateLimitedError:
                stats["failed"] += 1
                stats["failed_rate_limit"] += 1
                return
            except JiraAuthError:
                if stats["synced"] > 0 or stats["deleted"] > 0:
                    stats["failed"] += 1
                    stats["failed_auth"] += 1
                    return
                raise

            issues = payload.get("issues")
            if not isinstance(issues, list):
                raise JiraAPIError("Jira search response missing 'issues' list")

            for issue in issues:
                if not isinstance(issue, dict) or not issue.get("id"):
                    continue
                issue_id = str(issue["id"]).strip()
                fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
                updated_str = str(fields.get("updated") or "").strip()
                if updated_str and (newest_updated is None or updated_str > newest_updated):
                    newest_updated = updated_str

                current_run_ids.add(issue_id)
                known_ids.add(issue_id)
                row = _render_issue_row(client.base_url, project_key, issue)
                stats["synced"] += 1
                yield row

            total = int(payload.get("total") or 0)
            start_at += len(issues)
            if not issues or start_at >= total:
                break

        # If this was a full listing (no cursor filter), any previously known ID missing from
        # current_run_ids was deleted upstream in Jira -> emit tombstone.
        if cursor is None:
            removed_ids = sorted(known_ids - current_run_ids)
            for gone_id in removed_ids:
                known_ids.discard(gone_id)
                stats["deleted"] += 1
                yield {
                    "id": f"jira:issue:{gone_id}",
                    "title": "",
                    "content": "",
                    "url": "",
                    "_deleted": True,
                }

        p_state["cursor"] = newest_updated
        p_state["known_ids"] = sorted(known_ids)


def jira_source(
    *,
    base_url: str,
    project_keys: list[str] | tuple[str, ...],
    access_token: str | None = None,
    service: JiraClient | None = None,
    resource_name: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per Jira issue across selected projects."""
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Jira sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )

    norm_url = _normalize_base_url(base_url)
    keys = _normalize_project_keys(project_keys)

    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")
    if max_requests < 1:
        raise ValueError("max_requests must be at least 1.")

    if service is not None:
        client = service
    elif access_token is not None:
        client = build_jira_service(norm_url, access_token)
    else:
        raise ValueError("Either access_token or service must be provided.")

    if resource_name is None:
        digest = hashlib.sha256(f"{norm_url}|{'|'.join(keys)}".encode()).hexdigest()[:12]
        resource_name = f"jira_{digest}"
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
    def jira_issues():
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
            keys,
            dlt.current.resource_state(),
            stats,
            page_size=page_size,
            max_requests=max_requests,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = jira_issues()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "jira")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
