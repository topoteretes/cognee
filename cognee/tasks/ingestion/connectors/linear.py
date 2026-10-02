"""Linear connector for cognee: a ``dlt`` source that turns a team's work into memory.

One source is one Linear team. It yields the team's issues (with their comments
folded into the issue document) and its projects as flat document rows, and is
meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import linear_source

    await cognee.remember(
        linear_source(team_id="<team uuid>", access_token="<oauth token>"),
        dataset_name="linear",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted}``. Ids are prefixed by
  kind (``issue:<uuid>``, ``project:<uuid>``). Every column feeds the stored
  document identity, so the column set and the rendering are frozen: nothing
  volatile (timestamps of the last edit, relative dates) is rendered, and
  ``updatedAt`` lives only in resource state.
* **Incremental** by ``updatedAt``. Linear sorts newest first and has no change
  feed, so each stream (issues, comments, projects) keeps a floor, the ids seen
  exactly at the floor, and, while a walk is unfinished, a ceiling. A run with
  no changes yields nothing.
* **Comments** are part of their issue's document. A comment edit does not
  reliably bump ``Issue.updatedAt``, so changed comments are read as their own
  stream and the parent issues are re-rendered. A comment that was deleted
  is not seen, because Linear hard-deletes them.
* **A run is bounded and resumable.** On a rate limit, low remaining quota, a
  spent request budget or an expired token the source stops cleanly, keeps the
  state it reached and reports it in ``cognee_sync_stats`` (``failed_rate_limit``).
  It never raises or sleeps for these: ``dlt`` rolls the state of a failed run
  back, so a team too big for one hour of quota would never finish, and sleeping
  would hold the process-wide staging lock.
* **Token** is a plain access token. This source never refreshes it. It lives in
  the client object only; errors carry codes, never response bodies.

.. note::
   Archived issues are included (Linear archives closed issues automatically);
   trashed issues are skipped and stay in memory once ingested. Moving an issue
   to another team leaves it in the old team's table.

Privacy
-------
This connector reads the content of your Linear workspace. It is **opt-in**:
nothing is fetched until you construct a source and call ``remember``. Consent
covers private teams the token can see, and anyone with read access to the
target dataset can see what was ingested from them.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.linear.app/graphql"

# Linear caps a query at 10,000 complexity points and a connection multiplies its
# children by ``first``: 50 issues with up to 50 nested comments stay below it.
DEFAULT_PAGE_SIZE = 50
# 50 issues with 25 labels and 50 comments each cost about 8,000 of the 10,000 points
# Linear allows per query, so a bigger page could be refused outright.
MAX_PAGE_SIZE = 50
_COMMENT_PAGE_SIZE = 50
_DIRTY_CHUNK = 25
# Keep two maximum-size queries of headroom in the hourly complexity budget.
_COMPLEXITY_RESERVE = 20_000
_REQUESTS_RESERVE = 100
DEFAULT_MAX_REQUESTS = 1500
_CLOCK_MARGIN_SECONDS = 300
_TIMEOUT = (10, 30)
_RETRIES = 3

_ISSUE_FIELDS = """
id identifier title description url trashed createdAt updatedAt
state { name }
assignee { name }
priority
labels(first: 25) { nodes { name } }
project { name }
parent { identifier }
comments(first: __COMMENTS__) {
  pageInfo { hasNextPage endCursor }
  nodes { id body createdAt updatedAt user { name } }
}
""".replace("__COMMENTS__", str(_COMMENT_PAGE_SIZE))

_ISSUES_QUERY = (
    """
query LinearTeamIssues($teamId: String!, $filter: IssueFilter, $first: Int!, $after: String,
                       $archived: Boolean!) {
  team(id: $teamId) {
    issues(filter: $filter, first: $first, after: $after, orderBy: updatedAt,
           includeArchived: $archived) {
      pageInfo { hasNextPage endCursor }
      nodes { __ISSUE_FIELDS__ }
    }
  }
}
"""
).replace("__ISSUE_FIELDS__", _ISSUE_FIELDS)

_ISSUE_COMMENTS_QUERY = """
query LinearIssueComments($id: String!, $first: Int!, $after: String) {
  issue(id: $id) {
    comments(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { id body createdAt updatedAt user { name } }
    }
  }
}
"""

_COMMENTS_QUERY = """
query LinearTeamComments($filter: CommentFilter, $first: Int!, $after: String,
                         $archived: Boolean!) {
  comments(filter: $filter, first: $first, after: $after, orderBy: updatedAt,
           includeArchived: $archived) {
    pageInfo { hasNextPage endCursor }
    nodes { id updatedAt issue { id } }
  }
}
"""

_PROJECTS_QUERY = """
query LinearTeamProjects($teamId: String!, $filter: ProjectFilter, $first: Int!, $after: String,
                         $archived: Boolean!) {
  team(id: $teamId) {
    projects(filter: $filter, first: $first, after: $after, orderBy: updatedAt,
             includeArchived: $archived) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id name description content url startDate targetDate updatedAt
        status { name }
        lead { name }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Errors: messages carry codes only, never tokens, variables or response bodies.
# ---------------------------------------------------------------------------
class LinearSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class LinearAPIError(LinearSourceError):
    """Linear answered with an error, or could not be reached."""


class LinearAuthError(LinearSourceError):
    """Linear rejected the access token (HTTP 401)."""


class LinearRateLimitedError(LinearSourceError):
    """Linear answered HTTP 400 with ``extensions.code = RATELIMITED``."""

    def __init__(self, reset_at_ms: int | None = None):
        super().__init__("Linear rate limit exceeded (RATELIMITED)")
        self.reset_at_ms = reset_at_ms


class LinearEntityNotFoundError(LinearAPIError):
    """Linear answered that the requested entity does not exist or is not visible."""


class LinearTeamNotFoundError(LinearSourceError):
    """The team does not exist or the token can no longer see it."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def _int_header(headers: Any, name: str) -> int | None:
    try:
        return int(headers.get(name))
    except (TypeError, ValueError):
        return None


class LinearClient:
    """Minimal synchronous GraphQL client for Linear.

    ``execute`` returns the ``data`` dict. After every successful call
    ``rate_limit`` holds the remaining request and complexity budget Linear
    reported. Tests and hosts can pass any object with the same ``execute``
    method and ``rate_limit`` attribute instead.
    """

    def __init__(self, access_token: str, *, sleep: Callable[[float], None] = time.sleep):
        token = (access_token or "").strip()
        # Reject early and without echoing the value: requests would put an invalid
        # header value, token included, into its exception.
        if not token or not token.isascii() or not token.isprintable() or " " in token:
            raise ValueError("The Linear access token is empty or has an invalid format")
        self._access_token = token
        self._sleep = sleep
        self.rate_limit: dict[str, int | None] = {}

    def __repr__(self) -> str:
        return "LinearClient(access_token=<redacted>)"

    def _headers(self) -> dict[str, str]:
        token = self._access_token
        # Personal API keys are sent as is, OAuth tokens as Bearer.
        value = token if token.startswith("lin_api_") else f"Bearer {token}"
        return {"Authorization": value, "Content-Type": "application/json"}

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        import requests

        payload: dict[str, Any] = {"query": query, "variables": variables or {}}
        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.post(
                    GRAPHQL_URL, json=payload, headers=self._headers(), timeout=_TIMEOUT
                )
            except requests.RequestException as exc:
                if last:
                    raise LinearAPIError(f"Linear request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue
            if response.status_code >= 500 and not last:
                self._sleep(2**attempt)
                continue
            return self._parse(response)
        raise LinearAPIError("Linear request failed")  # pragma: no cover

    def _parse(self, response: Any) -> dict[str, Any]:
        status = response.status_code
        if status == 401:
            raise LinearAuthError("Linear rejected the access token (HTTP 401)")
        if status == 429:
            # Linear itself answers 400/RATELIMITED, but an edge in front of it may use 429.
            raise LinearRateLimitedError(
                _int_header(response.headers, "X-RateLimit-Requests-Reset")
            )
        try:
            body = response.json()
        except ValueError:
            body = {}
        codes = _error_codes(body)
        if "RATELIMITED" in codes:
            raise LinearRateLimitedError(
                _int_header(response.headers, "X-RateLimit-Requests-Reset")
            )
        if status != 200:
            raise LinearAPIError(f"Linear request failed: HTTP {status}")
        if _entity_not_found(body):
            raise LinearEntityNotFoundError("Linear entity not found or not accessible")
        if codes or (isinstance(body, dict) and body.get("errors")):
            detail = ", ".join(codes) if codes else "unknown"
            raise LinearAPIError(f"Linear request failed: GraphQL codes {detail}")
        self.rate_limit = {
            "requests": _int_header(response.headers, "X-RateLimit-Requests-Remaining"),
            "complexity": _int_header(response.headers, "X-RateLimit-Complexity-Remaining"),
        }
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise LinearAPIError("Linear response had no data")
        return data


def _entity_not_found(body: Any) -> bool:
    """Whether Linear reported a missing entity. Only matched, never echoed."""
    errors = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(errors, list):
        return False
    for error in errors:
        if not isinstance(error, dict):
            continue
        extensions = error.get("extensions")
        texts = [error.get("message")]
        if isinstance(extensions, dict):
            texts.append(extensions.get("userPresentableMessage"))
        if any(isinstance(t, str) and t.lower().startswith("entity not found") for t in texts):
            return True
    return False


def _error_codes(body: Any) -> list[str]:
    errors = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(errors, list):
        return []
    codes = set()
    for error in errors:
        if not isinstance(error, dict):
            continue
        extensions = error.get("extensions")
        code = extensions.get("code") if isinstance(extensions, dict) else None
        if isinstance(code, str) and code:
            codes.add(code)
    return sorted(codes)


def build_linear_service(access_token: str) -> LinearClient:
    """Wrap a core-owned access token in a client, the way Drive and Gmail do."""
    return LinearClient(access_token)


# ---------------------------------------------------------------------------
# Rendering: deterministic, no volatile fields.
# ---------------------------------------------------------------------------
_PRIORITIES = {0: "", 1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}


def _name(node: Any) -> str:
    return str(node.get("name") or "").strip() if isinstance(node, dict) else ""


def _lines(fields: list[tuple[str, str]]) -> list[str]:
    return [f"{label}: {value}" for label, value in fields if value]


def render_issue(issue: dict[str, Any], comments: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted(
        name
        for name in (_name(label) for label in (issue.get("labels") or {}).get("nodes") or [])
        if name
    )
    fields = [
        ("State", _name(issue.get("state"))),
        ("Assignee", _name(issue.get("assignee"))),
        ("Priority", _PRIORITIES.get(issue.get("priority") or 0, "")),
        ("Labels", ", ".join(labels)),
        ("Project", _name(issue.get("project"))),
        ("Parent", str((issue.get("parent") or {}).get("identifier") or "")),
    ]
    parts = _lines(fields)
    description = str(issue.get("description") or "").strip()
    if description:
        parts.extend(["", description])
    ordered = sorted(comments, key=lambda c: (str(c.get("createdAt") or ""), str(c.get("id"))))
    rendered = []
    for comment in ordered:
        body = str(comment.get("body") or "").strip()
        if not body:
            continue
        author = _name(comment.get("user")) or "Unknown"
        rendered.append(f"{author} ({str(comment.get('createdAt') or '')[:10]}): {body}")
    if rendered:
        parts.extend(["", "Comments:", *rendered])
    identifier = str(issue.get("identifier") or "").strip()
    title = str(issue.get("title") or "").strip() or "Untitled issue"
    return {
        "id": f"issue:{issue['id']}",
        "title": f"{identifier} {title}".strip(),
        "content": "\n".join(parts).strip() or title,
        "url": issue.get("url") or "",
        "_deleted": False,
    }


def render_project(project: dict[str, Any]) -> dict[str, Any]:
    fields = [
        ("Status", _name(project.get("status"))),
        ("Lead", _name(project.get("lead"))),
        ("Start", str(project.get("startDate") or "")),
        ("Target", str(project.get("targetDate") or "")),
    ]
    parts = _lines(fields)
    for key in ("description", "content"):
        text = str(project.get(key) or "").strip()
        if text:
            parts.extend(["", text])
    title = str(project.get("name") or "").strip() or "Untitled project"
    return {
        "id": f"project:{project['id']}",
        "title": title,
        "content": "\n".join(parts).strip() or title,
        "url": project.get("url") or "",
        "_deleted": False,
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _stamp(epoch_seconds: float) -> str:
    """Format like Linear's timestamps so the two compare as strings."""
    moment = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Walker:
    """One run of the three streams against a team, writing progress into ``state``."""

    def __init__(
        self,
        client: Any,
        team_id: str,
        state: dict,
        stats: dict[str, int],
        *,
        check_active: Callable[[], None] | None,
        include_archived: bool,
        page_size: int,
        max_requests: int,
    ):
        self.client = client
        self.team_id = team_id
        self.state = state
        self.stats = stats
        self.check_active = check_active
        self.include_archived = include_archived
        self.page_size = page_size
        self.max_requests = max_requests
        self.requests = 0
        self.emitted = 0
        self.emitted_issue_ids: set[str] = set()
        # comment id -> updatedAt as rendered into an issue this run
        self.rendered_comments: dict[str, str] = {}

    # -- requests ----------------------------------------------------------
    def _call(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if self.check_active is not None:
            self.check_active()
        self._check_budget()
        self.requests += 1
        try:
            return self.client.execute(query, variables)
        except LinearRateLimitedError:
            raise _CutShort("rate_limit") from None
        except LinearAuthError:
            if self.emitted:
                raise _CutShort("auth") from None
            raise

    def _check_budget(self) -> None:
        if self.requests >= self.max_requests:
            raise _CutShort("budget")
        remaining = getattr(self.client, "rate_limit", None) or {}
        requests_left, complexity_left = remaining.get("requests"), remaining.get("complexity")
        if requests_left is not None and requests_left < _REQUESTS_RESERVE:
            raise _CutShort("rate_limit")
        if complexity_left is not None and complexity_left < _COMPLEXITY_RESERVE:
            raise _CutShort("rate_limit")

    # -- one stream --------------------------------------------------------
    def _walk(
        self, name: str, fetch: Callable[[dict, str | None], tuple[list, dict]]
    ) -> Iterator[dict]:
        """Yield a stream's changed nodes, newest first, resumable after a cut."""
        stream = self.state.setdefault("streams", {}).setdefault(name, {})
        floor = stream.get("floor")
        at_floor = set(stream.get("at_floor") or [])
        ceiling = stream.get("ceiling")
        new_floor = stream.get("next_floor")
        new_at = set(stream.get("next_at") or [])

        window: dict[str, str] = {}
        if floor:
            window["gte"] = floor
        if ceiling:
            window["lte"] = ceiling
        filter_ = {"updatedAt": window} if window else {}

        # An unfinished walk resumes from the page cursor it stopped at, over the
        # exact filter it started with. Resuming by ceiling alone cannot get past
        # more rows sharing one timestamp than a run can read. If Linear refuses
        # the old cursor, fall back to the ceiling window.
        after = stream.get("resume_after")
        resume_filter = stream.get("resume_filter")
        if after and resume_filter is not None:
            filter_ = resume_filter
        else:
            after = None

        first_fetch = True
        while True:
            try:
                nodes, page_info = fetch(filter_, after)
            except LinearAPIError:
                if not (first_fetch and after):
                    raise
                after = None
                filter_ = {"updatedAt": window} if window else {}
                stream.pop("resume_after", None)
                stream.pop("resume_filter", None)
                nodes, page_info = fetch(filter_, after)
            first_fetch = False
            for node in nodes:
                stamp = str(node.get("updatedAt") or "")
                if new_floor is None or stamp > new_floor:
                    new_floor, new_at = stamp, {node["id"]}
                elif stamp == new_floor:
                    new_at.add(node["id"])
                stream["ceiling"] = stamp
                stream["next_floor"] = new_floor
                stream["next_at"] = sorted(new_at)
                if stamp == floor and node["id"] in at_floor:
                    self.stats["skipped"] += 1
                    continue
                self.stats["scanned"] += 1
                yield node
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor or cursor == after:
                # A walk that cannot move forward must not look finished: the
                # floor would advance past rows never read.
                raise LinearAPIError("Linear pagination did not advance")
            after = cursor
            stream["resume_after"] = after
            stream["resume_filter"] = filter_

        if new_floor is not None:
            stream["floor"] = new_floor
            stream["at_floor"] = sorted(new_at)
        for key in ("ceiling", "next_floor", "next_at", "resume_after", "resume_filter"):
            stream.pop(key, None)

    # -- fetchers ----------------------------------------------------------
    def _team_query(self, query: str, variables: dict[str, Any]) -> dict:
        try:
            data = self._call(query, {"teamId": self.team_id, **variables})
        except LinearEntityNotFoundError:
            raise LinearTeamNotFoundError("Linear team not found or not accessible") from None
        team = data.get("team")
        if not isinstance(team, dict):
            raise LinearTeamNotFoundError("Linear team not found or not accessible")
        return team

    def _fetch_issues(
        self, filter_: dict, after: str | None, first: int | None = None
    ) -> tuple[list, dict]:
        team = self._team_query(
            _ISSUES_QUERY,
            {
                "filter": filter_ or None,
                "first": first or self.page_size,
                "after": after,
                "archived": self.include_archived,
            },
        )
        connection = team.get("issues") or {}
        return connection.get("nodes") or [], connection.get("pageInfo") or {}

    def _fetch_comments(self, filter_: dict, after: str | None) -> tuple[list, dict]:
        scoped = {"issue": {"team": {"id": {"eq": self.team_id}}}, **filter_}
        data = self._call(
            _COMMENTS_QUERY,
            {
                "filter": scoped,
                "first": self.page_size,
                "after": after,
                "archived": self.include_archived,
            },
        )
        connection = data.get("comments") or {}
        return connection.get("nodes") or [], connection.get("pageInfo") or {}

    def _fetch_projects(self, filter_: dict, after: str | None) -> tuple[list, dict]:
        team = self._team_query(
            _PROJECTS_QUERY,
            {
                "filter": filter_ or None,
                "first": self.page_size,
                "after": after,
                "archived": self.include_archived,
            },
        )
        connection = team.get("projects") or {}
        return connection.get("nodes") or [], connection.get("pageInfo") or {}

    def _all_comments(self, issue: dict) -> list[dict]:
        connection = issue.get("comments") or {}
        comments = list(connection.get("nodes") or [])
        info = connection.get("pageInfo") or {}
        while info.get("hasNextPage"):
            cursor = info.get("endCursor")
            if not cursor:
                raise LinearAPIError("Linear pagination did not advance")
            try:
                data = self._call(
                    _ISSUE_COMMENTS_QUERY,
                    {"id": issue["id"], "first": _COMMENT_PAGE_SIZE, "after": cursor},
                )
            except LinearEntityNotFoundError:
                break  # the issue was deleted meanwhile; keep the comments read so far
            page = ((data.get("issue") or {}).get("comments")) or {}
            comments.extend(page.get("nodes") or [])
            info = page.get("pageInfo") or {}
            if info.get("hasNextPage") and info.get("endCursor") == cursor:
                raise LinearAPIError("Linear pagination did not advance")
        return comments

    def _issue_row(self, issue: dict) -> dict | None:
        if issue.get("trashed"):
            self.stats["skipped"] += 1
            return None
        self.emitted += 1
        self.emitted_issue_ids.add(issue["id"])
        comments = self._all_comments(issue)
        for comment in comments:
            self.rendered_comments[comment["id"]] = str(comment.get("updatedAt") or "")
        return render_issue(issue, comments)

    # -- the run -----------------------------------------------------------
    def rows(self) -> Iterator[dict]:
        # Projects are few and go first, so a team whose issues use the whole run
        # budget cannot starve them.
        for project in self._walk("projects", self._fetch_projects):
            self.emitted += 1
            yield render_project(project)

        # The first issue walk renders every comment that exists, so the comment
        # stream only needs what changed since this run began. The margin covers
        # clock skew; a re-read comment only re-renders an unchanged issue.
        self.state.setdefault("backfill_started", _stamp(time.time() - _CLOCK_MARGIN_SECONDS))
        for issue in self._walk("issues", self._fetch_issues):
            row = self._issue_row(issue)
            if row is not None:
                yield row

        comments_stream = self.state["streams"].setdefault("comments", {})
        if "floor" not in comments_stream and "ceiling" not in comments_stream:
            comments_stream["floor"] = self.state["backfill_started"]
            comments_stream["at_floor"] = []

        # A comment edit may not bump its issue, so changed comments are their
        # own stream and their parent issues are re-rendered. The pending ids
        # are kept in state so a cut run loses none of them. An issue rendered
        # earlier in this run already holds a comment only if it rendered that
        # comment at that edit: one edited or added since is queued again.
        pending = self.state.setdefault("pending_issue_ids", [])
        for comment in self._walk("comments", self._fetch_comments):
            issue_id = (comment.get("issue") or {}).get("id")
            if not issue_id or issue_id in pending:
                continue
            rendered = self.rendered_comments.get(comment["id"])
            if rendered is not None and rendered >= str(comment.get("updatedAt") or ""):
                continue
            pending.append(issue_id)
        while pending:
            chunk = pending[:_DIRTY_CHUNK]
            nodes, _ = self._fetch_issues({"id": {"in": chunk}}, None, first=len(chunk))
            for issue in nodes:
                row = self._issue_row(issue)
                if row is not None:
                    yield row
            del pending[: len(chunk)]


def _iter_rows(
    client: Any,
    team_id: str,
    state: dict,
    stats: dict[str, int],
    *,
    check_active: Callable[[], None] | None = None,
    include_archived: bool = True,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict]:
    """Yield the changed rows of a team. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        team_id,
        state,
        stats,
        check_active=check_active,
        include_archived=include_archived,
        page_size=page_size,
        max_requests=max_requests,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        stats["failed"] = max(1, stats.get("failed", 0))
        stats[f"failed_{cut.reason}"] = 1
        logger.warning("Linear sync stopped early (%s); it resumes on the next run.", cut.reason)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def _default_resource_name(team_id: str) -> str:
    return "linear_" + re.sub(r"[^a-z0-9]+", "_", team_id.lower()).strip("_")


def linear_source(
    *,
    team_id: str,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
    service: Any = None,
    access_token: str | None = None,
    include_archived: bool = True,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields one Linear team's documents for ``remember``.

    Args:
        team_id: The Linear team's id (the uuid, not its key).
        resource_name: Stable, connection-specific name. Must start with ``linear_``
            when a host tracks the table by that prefix. Defaults to one derived
            from the team id.
        check_active: Optional host authorization checkpoint, called before each
            request.
        service: Pre-built client (see :func:`build_linear_service`). Mainly an
            injection point for tests and hosts.
        access_token: Used to build the client when ``service`` is omitted.
        include_archived: Include archived issues (Linear archives closed issues
            automatically). Trashed issues are always skipped.
        page_size: Rows per page, at most 50: that is what keeps a page of issues
            with nested comments under Linear's 10,000-point query limit.
        max_requests: Request budget of one run. A bigger team finishes over
            several runs.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The Linear connector requires dlt (a cognee core dependency).") from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Linear sync requires a Cognee build with table-scoped DLT document cleanup."
        )
    if service is None and not access_token:
        raise ValueError("linear_source needs a service or an access_token")

    # Built here so a malformed token fails at construction, and the closure holds
    # the client, not the raw token.
    client = service if service is not None else LinearClient(access_token or "")
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    name = resource_name or _default_resource_name(team_id)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def linear_team():
        stats.clear()
        stats.update(scanned=0, skipped=0, failed=0)
        yield from _iter_rows(
            client,
            team_id,
            dlt.current.resource_state(),
            stats,
            check_active=check_active,
            include_archived=include_archived,
            page_size=page_size,
            max_requests=max_requests,
        )

    resource = linear_team()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "linear")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    resource.cognee_sync_stats = stats
    return resource
