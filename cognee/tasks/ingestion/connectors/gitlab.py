"""GitLab connector for cognee: a ``dlt`` source that turns a project's work into memory.

One source is one GitLab project. It yields the project's issues (with their discussion
notes folded into the issue document) and its merge requests as flat document rows, and is
meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import gitlab_source

    await cognee.remember(
        gitlab_source(project_id="my-group/my-project", access_token="<gitlab_token>"),
        dataset_name="gitlab_my_project",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted}``. Ids are prefixed by
  kind (``issue:<project_id>:<iid>``, ``mr:<project_id>:<iid>``). Every column feeds the stored
  document identity, so the column set and rendering are frozen: nothing
  volatile (relative dates, local timestamps) is rendered, and
  ``updated_at`` lives only in resource state.
* **Incremental** by ``updated_at``. GitLab supports keyset pagination and ``updated_after``.
  Each stream (issues, merge requests) maintains a floor timestamp, the ids seen
  exactly at the floor, and, while a walk is unfinished, a ceiling. A run with
  no changes yields nothing.
* **Discussion notes** are folded into their parent issue / MR document. System notes
  (e.g., automated state changes) are filtered out so that only real user discussions
  enrich the knowledge graph.
* **A run is bounded and resumable.** On a rate limit, low remaining quota, a
  spent request budget or an expired token, the source stops cleanly, keeps the
  state it reached and reports it in ``cognee_sync_stats`` (``failed_rate_limit``).
  It never raises or sleeps indefinitely for these: ``dlt`` rolls back the state of
  a failed run, so sleeping or raising would lose progress.
* **Token** supports both personal/project access tokens (``PRIVATE-TOKEN``) and
  OAuth bearer tokens (``Bearer``). Errors carry error codes and statuses,
  never tokens or sensitive URLs.
* **Self-hosted GitLab support**: Supports custom instances through ``base_url``
  (e.g., ``https://gitlab.example.com``), defaulting to ``https://gitlab.com``.

Privacy
-------
This connector reads the content of your GitLab project. It is **opt-in**:
nothing is fetched until you construct a source and call ``remember``. Consent
covers projects the token has access to, and anyone with read access to the
target dataset can see what was ingested from them.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote, urljoin

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

DEFAULT_GITLAB_URL = "https://gitlab.com"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
_DIRTY_CHUNK = 25
_RATE_LIMIT_RESERVE = 50
DEFAULT_MAX_REQUESTS = 1500
_TIMEOUT = (10, 30)
_RETRIES = 3


# ---------------------------------------------------------------------------
# Errors: messages carry codes and statuses only, never tokens or secrets.
# ---------------------------------------------------------------------------
class GitLabSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class GitLabAPIError(GitLabSourceError):
    """GitLab answered with an error, or could not be reached."""


class GitLabAuthError(GitLabSourceError):
    """GitLab rejected the access token (HTTP 401 / 403)."""


class GitLabRateLimitedError(GitLabSourceError):
    """GitLab rate limit exceeded (HTTP 429)."""

    def __init__(self, reset_at_epoch: int | None = None):
        super().__init__("GitLab rate limit exceeded (HTTP 429)")
        self.reset_at_epoch = reset_at_epoch


class GitLabNotFoundError(GitLabAPIError):
    """GitLab answered that the requested entity does not exist or is not visible."""


class GitLabProjectNotFoundError(GitLabSourceError):
    """The project does not exist or the token cannot see it."""


# ---------------------------------------------------------------------------
# Transport & Client
# ---------------------------------------------------------------------------
def _int_header(headers: Any, name: str) -> int | None:
    try:
        val = headers.get(name) if hasattr(headers, "get") else None
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


class GitLabClient:
    """Synchronous REST client for GitLab API v4.

    ``execute`` returns a tuple of ``(data, headers)``. After every call
    ``rate_limit`` holds the remaining request quota GitLab reported.
    Tests and hosts can pass any object with the same ``execute`` method
    and ``rate_limit`` attribute instead.
    """

    def __init__(
        self,
        access_token: str,
        *,
        base_url: str = DEFAULT_GITLAB_URL,
        private_token: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ):
        token = (access_token or "").strip()
        if not token or not token.isascii() or not token.isprintable() or " " in token:
            raise ValueError("The GitLab access token is empty or has an invalid format")

        clean_url = (base_url or DEFAULT_GITLAB_URL).strip().rstrip("/")
        if not clean_url.endswith("/api/v4"):
            clean_url = f"{clean_url}/api/v4"

        self._access_token = token
        self._base_url = clean_url
        self._private_token = private_token
        self._sleep = sleep
        self.rate_limit: dict[str, int | None] = {}

    def __repr__(self) -> str:
        return f"GitLabClient(base_url='{self._base_url}', access_token=<redacted>)"

    def _headers(self) -> dict[str, str]:
        if self._private_token:
            return {"PRIVATE-TOKEN": self._access_token, "Accept": "application/json"}
        return {"Authorization": f"Bearer {self._access_token}", "Accept": "application/json"}

    def execute(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        import requests

        clean_endpoint = endpoint.lstrip("/")
        url = f"{self._base_url}/{clean_endpoint}"
        query_params = {k: v for k, v in (params or {}).items() if v is not None}

        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.get(
                    url,
                    params=query_params,
                    headers=self._headers(),
                    timeout=_TIMEOUT,
                )
            except requests.RequestException as exc:
                if last:
                    raise GitLabAPIError(f"GitLab request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue

            # Update rate-limit info from headers
            remaining = _int_header(response.headers, "RateLimit-Remaining")
            reset_at = _int_header(response.headers, "RateLimit-Reset")
            self.rate_limit = {"remaining": remaining, "reset_at": reset_at}

            if response.status_code >= 500 and not last:
                self._sleep(2**attempt)
                continue

            return self._parse(response)

        raise GitLabAPIError("GitLab request failed")  # pragma: no cover

    def _parse(self, response: Any) -> tuple[Any, dict[str, Any]]:
        status = response.status_code
        if status in (401, 403):
            raise GitLabAuthError(f"GitLab rejected the token (HTTP {status})")
        if status == 404:
            raise GitLabNotFoundError(f"GitLab resource not found (HTTP 404)")
        if status == 429:
            reset_at = _int_header(response.headers, "RateLimit-Reset")
            if reset_at is None:
                retry_after = _int_header(response.headers, "Retry-After")
                if retry_after is not None:
                    reset_at = int(time.time()) + retry_after
            raise GitLabRateLimitedError(reset_at_epoch=reset_at)
        if status >= 400:
            raise GitLabAPIError(f"GitLab API request failed with HTTP {status}")

        try:
            data = response.json()
        except Exception as exc:
            raise GitLabAPIError("GitLab returned invalid JSON") from exc

        return data, dict(response.headers)


# ---------------------------------------------------------------------------
# Document Renderers (Deterministic text without dynamic/relative dates)
# ---------------------------------------------------------------------------
def _format_author(user_obj: dict | None) -> str:
    if not isinstance(user_obj, dict):
        return "Unknown"
    name = (user_obj.get("name") or "").strip()
    username = (user_obj.get("username") or "").strip()
    if name and username:
        return f"{name} (@{username})"
    return name or username or "Unknown"


def render_issue(issue: dict[str, Any], notes: list[dict[str, Any]], project_path: str) -> str:
    """Render a GitLab issue and its discussion comments deterministically."""
    iid = issue.get("iid", "")
    title = (issue.get("title") or "").strip()
    state = issue.get("state", "opened")
    web_url = issue.get("web_url", "")
    author = _format_author(issue.get("author"))

    assignees_list = issue.get("assignees") or []
    assignees = ", ".join(_format_author(u) for u in assignees_list) if assignees_list else "None"

    milestone_obj = issue.get("milestone")
    milestone = milestone_obj.get("title") if isinstance(milestone_obj, dict) else "None"

    labels_list = issue.get("labels") or []
    labels = ", ".join(labels_list) if labels_list else "None"

    description = (issue.get("description") or "").strip()
    if not description:
        description = "*No description provided.*"

    parts = [
        f"# [Issue #{iid}] {title}",
        f"- **Project**: {project_path}",
        f"- **Type**: GitLab Issue",
        f"- **State**: {state}",
        f"- **Author**: {author}",
        f"- **Assignees**: {assignees}",
        f"- **Milestone**: {milestone}",
        f"- **Labels**: {labels}",
        f"- **URL**: {web_url}",
        "",
        "## Description",
        description,
    ]

    # Fold discussion notes deterministically (sorted by created_at and id)
    valid_notes = [
        note for note in notes
        if isinstance(note, dict) and not note.get("system", False) and (note.get("body") or "").strip()
    ]
    valid_notes.sort(key=lambda n: (str(n.get("created_at", "")), int(n.get("id", 0))))

    if valid_notes:
        parts.append("")
        parts.append(f"## Discussion ({len(valid_notes)} comments)")
        for note in valid_notes:
            note_author = _format_author(note.get("author"))
            note_body = (note.get("body") or "").strip()
            parts.append(f"### Comment by {note_author}")
            parts.append(note_body)
            parts.append("")

    return "\n".join(parts).strip()


def render_merge_request(mr: dict[str, Any], notes: list[dict[str, Any]], project_path: str) -> str:
    """Render a GitLab merge request and its discussion comments deterministically."""
    iid = mr.get("iid", "")
    title = (mr.get("title") or "").strip()
    state = mr.get("state", "opened")
    web_url = mr.get("web_url", "")
    author = _format_author(mr.get("author"))
    source_branch = mr.get("source_branch", "")
    target_branch = mr.get("target_branch", "")
    draft = "Yes" if mr.get("draft") or mr.get("work_in_progress") else "No"

    reviewers_list = mr.get("reviewers") or []
    reviewers = ", ".join(_format_author(u) for u in reviewers_list) if reviewers_list else "None"

    labels_list = mr.get("labels") or []
    labels = ", ".join(labels_list) if labels_list else "None"

    description = (mr.get("description") or "").strip()
    if not description:
        description = "*No description provided.*"

    parts = [
        f"# [MR !{iid}] {title}",
        f"- **Project**: {project_path}",
        f"- **Type**: GitLab Merge Request",
        f"- **State**: {state}",
        f"- **Draft**: {draft}",
        f"- **Source Branch**: {source_branch}",
        f"- **Target Branch**: {target_branch}",
        f"- **Author**: {author}",
        f"- **Reviewers**: {reviewers}",
        f"- **Labels**: {labels}",
        f"- **URL**: {web_url}",
        "",
        "## Description",
        description,
    ]

    valid_notes = [
        note for note in notes
        if isinstance(note, dict) and not note.get("system", False) and (note.get("body") or "").strip()
    ]
    valid_notes.sort(key=lambda n: (str(n.get("created_at", "")), int(n.get("id", 0))))

    if valid_notes:
        parts.append("")
        parts.append(f"## Discussion ({len(valid_notes)} comments)")
        for note in valid_notes:
            note_author = _format_author(note.get("author"))
            note_body = (note.get("body") or "").strip()
            parts.append(f"### Comment by {note_author}")
            parts.append(note_body)
            parts.append("")

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Walker: Bounded, Resumable Crawl over Issues and Merge Requests
# ---------------------------------------------------------------------------
class _CutShort(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class _GitLabWalker:
    def __init__(
        self,
        client: Any,
        project_id: str | int,
        state: dict[str, Any],
        stats: dict[str, int],
        *,
        check_active: Callable[[], None] | None = None,
        include_merge_requests: bool = True,
        include_closed: bool = True,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_requests: int = DEFAULT_MAX_REQUESTS,
    ):
        self.client = client
        self.project_id = str(project_id)
        self.encoded_project_id = quote(str(project_id), safe="")
        self.state = state
        self.stats = stats
        self.check_active = check_active
        self.include_merge_requests = include_merge_requests
        self.include_closed = include_closed
        self.page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
        self.max_requests = max_requests
        self._request_count = 0
        self.project_path_with_namespace = self.project_id

    def _call(self, endpoint: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if self.check_active:
            self.check_active()
        self._check_budget()
        self._request_count += 1
        return self.client.execute(endpoint, params=params)

    def _check_budget(self) -> None:
        if self._request_count >= self.max_requests:
            raise _CutShort("request_budget")

        remaining = self.client.rate_limit.get("remaining")
        if remaining is not None and remaining <= _RATE_LIMIT_RESERVE:
            raise _CutShort("rate_limit")

    def _init_project(self) -> None:
        try:
            project_data, _ = self._call(f"projects/{self.encoded_project_id}")
            if isinstance(project_data, dict):
                self.project_path_with_namespace = project_data.get("path_with_namespace") or self.project_id
        except GitLabNotFoundError:
            raise GitLabProjectNotFoundError(f"GitLab project '{self.project_id}' not found") from None
        except GitLabAPIError:
            pass  # Fall back to self.project_id

    def rows(self) -> Iterator[dict[str, Any]]:
        self._init_project()

        # 1. Walk Issues
        yield from self._walk_stream(
            name="issues",
            fetch_items=self._fetch_issues,
            fetch_notes=self._fetch_issue_notes,
            render_fn=render_issue,
            id_prefix=f"issue:{self.project_id}:",
        )

        # 2. Walk Merge Requests
        if self.include_merge_requests:
            yield from self._walk_stream(
                name="merge_requests",
                fetch_items=self._fetch_merge_requests,
                fetch_notes=self._fetch_mr_notes,
                render_fn=render_merge_request,
                id_prefix=f"mr:{self.project_id}:",
            )

    def _walk_stream(
        self,
        name: str,
        fetch_items: Callable[[str | None, int], tuple[list[dict], dict]],
        fetch_notes: Callable[[int], list[dict]],
        render_fn: Callable[[dict, list[dict], str], str],
        id_prefix: str,
    ) -> Iterator[dict[str, Any]]:
        stream = self.state.setdefault("streams", {}).setdefault(name, {})
        floor = stream.get("floor")
        at_floor = set(stream.get("at_floor") or [])
        ceiling = stream.get("ceiling")
        new_floor = stream.get("next_floor")
        new_at = set(stream.get("next_at") or [])

        page = stream.get("page", 1)
        updated_after = floor

        while True:
            items, headers = fetch_items(updated_after, page)
            if not items:
                break

            for item in items:
                item_id = str(item.get("id"))
                item_iid = item.get("iid", item_id)
                stamp = str(item.get("updated_at") or "")

                if new_floor is None or stamp > new_floor:
                    new_floor, new_at = stamp, {item_id}
                elif stamp == new_floor:
                    new_at.add(item_id)

                stream["ceiling"] = stamp
                stream["next_floor"] = new_floor
                stream["next_at"] = sorted(new_at)

                if stamp == floor and item_id in at_floor:
                    self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                    continue

                notes = fetch_notes(item_iid)
                rendered_content = render_fn(item, notes, self.project_path_with_namespace)

                doc_id = f"{id_prefix}{item_iid}"
                self.stats["scanned"] = self.stats.get("scanned", 0) + 1

                yield {
                    "id": doc_id,
                    "title": item.get("title", ""),
                    "content": rendered_content,
                    "url": item.get("web_url", ""),
                    "_deleted": False,
                    dlt_utils.NODE_SET_COLUMN: [f"gitlab:{self.project_path_with_namespace}"],
                }

            # Check next page from GitLab headers
            next_page = headers.get("X-Next-Page")
            if not next_page or not str(next_page).strip():
                break

            page = int(next_page)
            stream["page"] = page

        if new_floor is not None:
            stream["floor"] = new_floor
            stream["at_floor"] = sorted(new_at)

        for key in ("ceiling", "next_floor", "next_at", "page"):
            stream.pop(key, None)

    def _fetch_issues(self, updated_after: str | None, page: int) -> tuple[list[dict], dict]:
        params = {
            "order_by": "updated_at",
            "sort": "asc",
            "per_page": self.page_size,
            "page": page,
        }
        if updated_after:
            params["updated_after"] = updated_after
        if not self.include_closed:
            params["state"] = "opened"

        endpoint = f"projects/{self.encoded_project_id}/issues"
        data, headers = self._call(endpoint, params=params)
        return data if isinstance(data, list) else [], headers

    def _fetch_issue_notes(self, issue_iid: int | str) -> list[dict]:
        endpoint = f"projects/{self.encoded_project_id}/issues/{issue_iid}/notes"
        params = {"sort": "asc", "per_page": 100}
        try:
            data, _ = self._call(endpoint, params=params)
            return data if isinstance(data, list) else []
        except GitLabAPIError:
            return []

    def _fetch_merge_requests(self, updated_after: str | None, page: int) -> tuple[list[dict], dict]:
        params = {
            "order_by": "updated_at",
            "sort": "asc",
            "per_page": self.page_size,
            "page": page,
        }
        if updated_after:
            params["updated_after"] = updated_after
        if not self.include_closed:
            params["state"] = "opened"

        endpoint = f"projects/{self.encoded_project_id}/merge_requests"
        data, headers = self._call(endpoint, params=params)
        return data if isinstance(data, list) else [], headers

    def _fetch_mr_notes(self, mr_iid: int | str) -> list[dict]:
        endpoint = f"projects/{self.encoded_project_id}/merge_requests/{mr_iid}/notes"
        params = {"sort": "asc", "per_page": 100}
        try:
            data, _ = self._call(endpoint, params=params)
            return data if isinstance(data, list) else []
        except GitLabAPIError:
            return []


def _iter_rows(
    client: Any,
    project_id: str | int,
    state: dict[str, Any],
    stats: dict[str, int],
    *,
    check_active: Callable[[], None] | None = None,
    include_merge_requests: bool = True,
    include_closed: bool = True,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    stats.setdefault("scanned", 0)
    stats.setdefault("skipped", 0)
    stats.setdefault("failed", 0)
    walker = _GitLabWalker(
        client,
        project_id,
        state,
        stats,
        check_active=check_active,
        include_merge_requests=include_merge_requests,
        include_closed=include_closed,
        page_size=page_size,
        max_requests=max_requests,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        stats["failed"] = max(1, stats.get("failed", 0))
        stats[f"failed_{cut.reason}"] = 1
        logger.warning("GitLab sync stopped early (%s); it resumes on the next run.", cut.reason)


# ---------------------------------------------------------------------------
# Public Factories
# ---------------------------------------------------------------------------
def _default_resource_name(project_id: str | int) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(project_id).lower()).strip("_")
    return f"gitlab_{cleaned}"


def build_gitlab_service(
    access_token: str,
    *,
    base_url: str = DEFAULT_GITLAB_URL,
    private_token: bool = True,
) -> GitLabClient:
    """Build a standard GitLab API client."""
    return GitLabClient(access_token, base_url=base_url, private_token=private_token)


def gitlab_source(
    *,
    project_id: str | int,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
    service: Any = None,
    access_token: str | None = None,
    base_url: str = DEFAULT_GITLAB_URL,
    private_token: bool = True,
    include_merge_requests: bool = True,
    include_closed: bool = True,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields one GitLab project's documents for ``remember``.

    Args:
        project_id: The GitLab project ID (integer) or URL-encoded path (e.g. ``my-group/my-project``).
        resource_name: Stable, connection-specific name. Defaults to ``gitlab_<project_id>``.
        check_active: Optional host authorization checkpoint, called before each request.
        service: Pre-built client (see :func:`build_gitlab_service`). Injected for tests/hosts.
        access_token: Token string used to build the client when ``service`` is omitted.
        base_url: URL of the GitLab instance (default: ``https://gitlab.com``).
        private_token: True to send as ``PRIVATE-TOKEN`` (personal/project access token),
            False for OAuth ``Bearer`` token.
        include_merge_requests: Whether to ingest merge requests alongside issues.
        include_closed: Include closed issues and merged/closed MRs.
        page_size: Number of items per page (1 to 100).
        max_requests: Maximum number of HTTP requests allowed in one sync run.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    try:
        import dlt  # type: ignore
    except ImportError as exc:
        raise ImportError("The GitLab connector requires dlt (a cognee core dependency).") from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "GitLab sync requires a Cognee build with table-scoped DLT document cleanup."
        )

    if service is None and not access_token:
        raise ValueError("gitlab_source needs a service or an access_token")

    client = service if service is not None else GitLabClient(
        access_token or "",
        base_url=base_url,
        private_token=private_token,
    )
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    name = resource_name or _default_resource_name(project_id)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            **getattr(dlt_utils, "NODE_SET_COLUMN_HINT", {}),
        },
    )
    def gitlab_project():
        stats.clear()
        stats.update(scanned=0, skipped=0, failed=0)
        yield from _iter_rows(
            client,
            project_id,
            dlt.current.resource_state(),
            stats,
            check_active=check_active,
            include_merge_requests=include_merge_requests,
            include_closed=include_closed,
            page_size=page_size,
            max_requests=max_requests,
        )

    resource = gitlab_project()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "gitlab")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    resource.cognee_sync_stats = stats
    return resource
