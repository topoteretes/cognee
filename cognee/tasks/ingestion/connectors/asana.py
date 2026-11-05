# Adapted from topoteretes/cognee-community (Apache-2.0), PR #349, commit ef32699 (AbdulWasih05).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Asana connector for cognee: a ``dlt`` source that turns selected projects into memory.

One source is one Asana access token and a selection of projects. It yields one document per
task (with its comments and subtasks folded in) and one per selected project as flat rows, and
is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import asana_source

    await cognee.remember(
        asana_source(
            resource_name="asana_website",
            access_token="<personal access token>",
            project_gids=["1201234567890123"],
        ),
        dataset_name="asana",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are prefixed
  by kind (``task:<gid>``, ``project:<gid>``). Every column feeds the stored document identity,
  so the rendering is frozen and deterministic: no timestamps other than the day a task was
  completed, no emails, every collection sorted. A tombstone carries no node set.
* **Node set** ``asana:<project gid>`` per selected project a task is in (sorted), and
  ``asana:<its own gid>`` for a project row, so a ``recall`` can be scoped to one project. Only
  gids are used: renaming a project changes no node set. It is hashed into the document
  identity, so a task that joins or leaves a selected project is extracted once more.
* **Selection** is ``project_gids``, required. A task in two selected projects is one document
  and names only the selected projects. A workspace-wide sync is a caller listing its projects.
* **Incremental** in two parts. Every pass lists each selected project's tasks with their
  ``modified_at``; the listing decides which tasks changed (a window that starts a minute before
  the stored cursor, and ``seen`` to tell a rendered task from one that moved). What
  ``modified_at`` does not show (a comment edit, a subtask change, a rename of a project or a
  section) arrives through the Events API, kept as one sync token per project. An expired token
  (HTTP 412) re-renders the project.
* **Forget-on-delete.** Only a complete listing forgets, and a task it lacks is forgotten only
  when Asana confirms it is gone: a known task missing from the listing is asked for its
  memberships (one request, counted in ``max_requests``), and a 403, a 404 or no selected
  project forgets it, while a selected project keeps it (a listing can miss a live task that
  moved between projects, or on a page that shifted), and so does an answer of 5xx or nothing,
  which is unknown, not gone. Each check settles its own task, so a run cut inside the listing,
  or failing, forgets nothing, while a run cut between two checks keeps the tombstones already
  emitted. Deselecting a project forgets the tasks that were only in it, one check each.
* **A project Asana will not show** (403 or 404 on its own read) is unresolved: skipped, tried
  again every run, counted in ``cognee_sync_stats["unresolved"]``. Its stored cursor, sync
  token, ``seen`` and task gids stay, its row, its tasks and the tasks of it that were queued
  for a read or a retry are kept, and every other project syncs. A 403 and a 404 are told apart
  in the log and treated alike for everything else.
* **A task that cannot be read** (403, 404, 5xx or no answer after the retries, or a 200 of the
  wrong shape in the task, its comments or its subtasks) while it is still listed keeps its old
  document and is asked again on the next run, after every fresh task. It is never tombstoned
  and never blocks another task. A 200 of the wrong shape in a listing, the events feed, a
  project read or the check of a missing task raises instead.
* **A run is bounded and resumable.** On a rate limit, a spent request budget, an outage (three
  tasks in a row answering 5xx or nothing, counted apart in the fresh tasks, in the retry tail
  and in the checks of missing tasks) or a rejected token after rows were emitted, the source stops cleanly, keeps what it
  emitted, queues the tasks it did not reach in ``state["pending"]`` and reports the reason in
  ``cognee_sync_stats`` (``failed_rate_limit``, ``failed_budget``, ``failed_unavailable``,
  ``failed_auth``). It never sleeps on a 429: ``dlt`` rolls back the state of a failed run and
  the walk holds the process-wide staging lock.
* **The listing is a pass that can span runs.** A project whose listing finished, and the events
  and page offset of one cut inside, are kept in ``state["listing"]`` (ids, timestamps and
  offsets, never text) and the finished projects' rows are emitted at once. A run cut inside the
  listing (a 429 or a spent budget) renders nothing and forgets nothing; the next run continues
  at the page it stopped at, and the pass completes, renders and forgets only when every
  selected project is listed. A changed selection starts the pass over.
* **Budget.** ``max_requests`` bounds every request of a run, the listing's and the reads' (a
  task costs three, so at least three), so a run holds the staging lock for at most that many
  requests plus the client's retries of each (``_RETRIES``); there is no deadline or other
  limit. The listing costs ``projects x (1 + events + ceil(tasks / 100))`` requests a pass, and
  rendering waits for a complete pass. The budget must cover the re-read of every unresolved
  project plus one task read or one check; a run cut by it that kept nothing raises
  ``AsanaSourceError``. Retries are bounded by ``_RETRIES`` alone.
* **Token** is a plain access token (personal access token or an OAuth access token). It lives
  in the client object only; errors carry statuses and endpoints, never the token, a task id or
  Asana's own explanation (that is an attribute of the error, and is not logged).

Privacy
-------
This connector reads task names, descriptions, comments, assignee and author names and custom
fields of the selected projects. It is **opt-in**: nothing is fetched until you construct a
source and call ``remember``. What it reads goes to your configured LLM and embedding
providers, and anyone with read access to the target dataset can see it. Email addresses are
never requested, but people type them into comments and descriptions.
"""

import copy
import re
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = get_logger("asana_connector")

BASE_URL = "https://app.asana.com/api/1.0"
# Asana's largest ``limit`` on the list endpoints used here.
PAGE_SIZE = 100
# Budget of the requests of one run, the listing's and the reads'. Our own choice, not an API
# limit: Asana documents only per-minute limits. A task costs three requests.
DEFAULT_MAX_REQUESTS = 1500
# Version of the resource state; a later release reads it to migrate an older layout.
STATE_VERSION = 1
# (connect, read) seconds; our own choice, Asana documents no expected latency.
_TIMEOUT = (10, 30)
# Attempts per request on 5xx or network errors, backing off 1s then 2s.
_RETRIES = 3
# How far before the cursor the listing window starts: a task can carry the cursor's timestamp
# or one just before it and still have been missed by the previous run.
_OVERLAP_SECONDS = 60
# Fresh tasks in a row answering 5xx or nothing that mean Asana is down. One or two can be a
# broken task, three in a row rarely are, and a stop costs nothing: the next run resumes.
_OUTAGE_STREAK = 3
# Requests that read one task: the task, its comments and its subtasks.
_TASK_REQUESTS = 3
# A gid goes into the URL of its own read, so no ``.``, ``/``, ``?``, ``#`` or ``%`` may get
# through. Asana's gids are digits; only the character set is checked.
_GID = re.compile(r"[0-9A-Za-z]+")

# Asana returns compact objects unless fields are requested, so each call asks only for what
# is rendered (plus created_at on stories, to put comments in order).
_TASK_FIELDS = (
    "name,notes,completed,completed_at,due_on,permalink_url,assignee.name,"
    "memberships.project.gid,memberships.project.name,memberships.section.name,"
    "custom_fields.name,custom_fields.display_value"
)
# The custom field keys that must be text (or null) in an answer.
_FIELD_TEXTS = ("name", "display_value")
_STORY_FIELDS = "resource_subtype,text,created_by.name,created_at"
_SUBTASK_FIELDS = "name,completed,assignee.name,due_on,notes"
_PROJECT_FIELDS = "name,notes,permalink_url"
# resource.target names the task of an edited comment; change.field tells a rename from other
# edits (see _event_task_gids and _renamed).
_EVENT_FIELDS = (
    "action,parent.gid,parent.resource_type,resource.gid,resource.resource_type,"
    "resource.target.gid,resource.target.resource_type,change.field"
)


# ---------------------------------------------------------------------------
# Errors: messages carry statuses and endpoints only, never tokens, ids or bodies.
# ---------------------------------------------------------------------------
class AsanaSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class AsanaAPIError(AsanaSourceError):
    """An error answer from Asana (``status``), or no answer at all (``status`` 0).

    The message names the status and the endpoint with ids replaced. Asana's own explanation is
    on ``detail`` with the token cut out; it is for the caller to read and is never logged.
    """

    def __init__(self, status: int, endpoint: str, detail: str = ""):
        answer = f"HTTP {status}" if status else "no answer"
        super().__init__(f"Asana {endpoint}: {answer}")
        self.status = status
        self.endpoint = endpoint
        self.detail = detail


class AsanaAuthError(AsanaAPIError):
    """Asana rejected the access token (HTTP 401)."""

    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(401, endpoint, detail)


class AsanaRateLimitedError(AsanaAPIError):
    """Asana answered HTTP 429."""

    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(429, endpoint, detail)


class AsanaSyncTokenExpiredError(AsanaAPIError):
    """``/events`` answered HTTP 412: the sync token is missing or too old.

    Asana sends a fresh one with the answer, ``sync_token``.
    """

    def __init__(self, endpoint: str, sync_token: str):
        super().__init__(412, endpoint)
        self.sync_token = sync_token


class AsanaResponseError(AsanaSourceError):
    """Asana answered ``200`` with a body that is not of the expected shape.

    Never read as an empty listing or a skippable task. In a listing, the events feed, a project
    read or a check of a missing task it raises and aborts the run. In the read of one task (the
    task, its comments or its subtasks) the task is kept for a retry and the run goes on.
    """


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def _endpoint(path: str) -> str:
    """The path with a task or project gid replaced, so messages never carry one."""
    return re.sub(r"^/(tasks|projects)/[^/]+", r"/\1/{gid}", path)


class AsanaClient:
    """Minimal synchronous REST client for Asana.

    ``request`` returns the JSON object of a ``200`` answer. Tests and hosts can pass any
    object with the same ``request`` method instead, one that takes ``check_active`` and calls it
    before each attempt it makes.
    """

    def __init__(self, access_token: str, *, sleep: Callable[[float], None] = time.sleep):
        # This check is what keeps the token out of an exception: requests names the value of a
        # header it rejects, and every token it would reject is refused here without echoing it.
        token = access_token.strip() if isinstance(access_token, str) else ""
        if (
            not token
            or not token.isascii()
            or not token.isprintable()
            or any(char.isspace() for char in token)
        ):
            raise ValueError("The Asana access token is empty or has an invalid format")
        self._access_token = token
        self._sleep = sleep

    def __repr__(self) -> str:
        return "AsanaClient(access_token=<redacted>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
    ) -> dict:
        """The JSON object of a ``200`` answer, retrying a 5xx or a network error.

        ``check_active`` is called before each attempt, a retry included, and before its backoff
        sleep, so a host that has revoked the run does not wait out the retries. What it raises
        propagates as it is: it is no network error, so it is neither retried nor wrapped.
        """
        import requests  # lazy: importing the connectors package loads no client library

        headers = {"Authorization": f"Bearer {self._access_token}", "Accept": "application/json"}
        response = None
        for attempt in range(_RETRIES):
            if check_active is not None:
                check_active()
            if attempt:
                self._sleep(2 ** (attempt - 1))
            try:
                response = requests.request(
                    method, BASE_URL + path, params=params, headers=headers, timeout=_TIMEOUT
                )
            except requests.RequestException:
                # Only the fact: the message of some requests errors holds a header.
                response = None
                continue
            if response.status_code < 500:
                break
        if response is None:
            raise AsanaAPIError(0, _endpoint(path))
        return self._parse(response, path)

    def _parse(self, response: Any, path: str) -> dict:
        status, endpoint = response.status_code, _endpoint(path)
        try:
            body = response.json()
        except ValueError:
            body = None
        if status == 401:
            raise AsanaAuthError(endpoint, self._detail(body))
        if status == 429:
            raise AsanaRateLimitedError(endpoint, self._detail(body))
        if status == 412 and isinstance(body, dict) and isinstance(body.get("sync"), str):
            raise AsanaSyncTokenExpiredError(endpoint, body["sync"])
        if status != 200:
            raise AsanaAPIError(status, endpoint, self._detail(body))
        if not isinstance(body, dict):
            raise AsanaResponseError(f"Asana {endpoint}: HTTP 200 without a JSON object")
        return body

    def _detail(self, body: Any) -> str:
        """Asana's ``errors[0].message`` with the token cut out, else empty. Never the raw body.

        The token is cut out before the text is shortened, so a cut can never leave a prefix of it.
        """
        errors = body.get("errors") if isinstance(body, dict) else None
        first = errors[0] if isinstance(errors, list) and errors else None
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, str):
            return ""
        return message.replace(self._access_token, "[token]")[:200]


def build_asana_service(access_token: str) -> AsanaClient:
    """Wrap a host-owned access token in a client, the way Linear and Apollo do.

    A ``service`` handed to :func:`asana_source` instead must offer
    ``request(method, path, *, params, check_active)`` and call ``check_active`` before each
    attempt it makes.
    """
    return AsanaClient(access_token)


def _data(body: dict, shape: type, endpoint: str) -> Any:
    """``body["data"]``, which must be a ``list`` or a ``dict`` as expected."""
    data = body.get("data")
    if not isinstance(data, shape):
        raise AsanaResponseError(
            f"Asana {endpoint}: data is a {type(data).__name__}, not a {shape.__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def _instant(stamp: str) -> datetime:
    """Parse a timestamp such as ``2026-10-09T19:14:27.123Z`` (3.10 reads no ``Z``)."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _stamp(moment: datetime) -> str:
    """``moment`` in Asana's format: UTC, three decimals, ``Z`` (``2026-10-09T19:14:27.123Z``).

    Timestamps are compared and stored as strings, which orders them only in this one form, so
    every ``modified_at`` goes through here (``_stamp(_instant(value))``) before it is used.
    A moment without an offset raises ``ValueError``: it has no UTC instant, and ``astimezone``
    would read it as the machine's local time.
    """
    if moment.tzinfo is None:
        raise ValueError("a timestamp without an offset")
    moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _overlap_start(cursor: str) -> str:
    """The timestamp ``_OVERLAP_SECONDS`` before ``cursor``, in Asana's format; empty for none."""
    if not cursor:
        return ""
    return _stamp(_instant(cursor) - timedelta(seconds=_OVERLAP_SECONDS))


# ---------------------------------------------------------------------------
# Events -> tasks
# ---------------------------------------------------------------------------
def _ref(value: Any) -> dict:
    """``value`` when it is an object, else an empty one: events carry strings and nulls too."""
    return value if isinstance(value, dict) else {}


def _event_task_gids(events: Iterable[Any]) -> set[str]:
    """The gids of the tasks a batch of project events is about.

    * A story (comment) event names its task as ``parent`` when the comment was added or
      removed; when it is *edited*, only ``resource.target`` does.
    * A task event whose ``parent`` is a task means a subtask was added to or removed from that
      parent, so the parent is the one to re-render.
    * Any other task event is about the task named in ``resource``.

    Events without a gid are skipped. The result may hold subtask gids and gids of deleted
    tasks; ``_documents_for`` maps or drops them.
    """
    gids: set[str] = set()
    for event in events:
        resource, parent = _ref(_ref(event).get("resource")), _ref(_ref(event).get("parent"))
        kind = resource.get("resource_type")
        owner: dict = {}
        if kind == "story":
            owner = (
                parent if parent.get("resource_type") == "task" else _ref(resource.get("target"))
            )
            if owner.get("resource_type") != "task":
                owner = {}
        elif kind == "task":
            owner = parent if parent.get("resource_type") == "task" else resource
        gid = owner.get("gid")
        if isinstance(gid, str) and gid:
            gids.add(gid)
    return gids


def _renamed(events: Iterable[Any]) -> bool:
    """Whether the project or one of its sections was renamed.

    Every task document carries its project and section names, and a rename bumps no task's
    ``modified_at``, so the project's tasks are re-rendered.
    """
    return any(
        _ref(_ref(event).get("resource")).get("resource_type") in ("project", "section")
        and _ref(event).get("action") == "changed"
        and _ref(_ref(event).get("change")).get("field") == "name"
        for event in events
    )


def _documents_for(
    task_gids: set[str], swept: set[str], subtask_parents: dict[str, str]
) -> set[str]:
    """Map event task gids onto the task documents they affect.

    A gid in ``swept`` is a document itself. Otherwise it may be a subtask whose title is folded
    into its parent's document; ``subtask_parents`` remembers which, because renaming or deleting
    a subtask touches neither the parent nor the event's ``parent`` field. Anything else drops.
    """
    documents: set[str] = set()
    for gid in task_gids:
        if gid in swept:
            documents.add(gid)
        elif subtask_parents.get(gid) in swept:
            documents.add(subtask_parents[gid])
    return documents


# ---------------------------------------------------------------------------
# Rendering: deterministic, no volatile fields, no emails.
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _name(node: Any) -> str:
    return _text(node.get("name")) if isinstance(node, dict) else ""


def _gid_order(item: dict) -> tuple[int, str]:
    """Creation order of Asana objects: gids are numeric strings that grow over time."""
    gid = str(item.get("gid") or "")
    return len(gid), gid


def _memberships(task: dict) -> list[dict]:
    """The memberships of a task that are in a project, checked.

    An entry with no ``project`` (absent or null) is not a project membership, for example a
    task on someone's My Tasks list, and is skipped. A malformed answer raises
    ``AsanaResponseError``: ``memberships`` that is not a list, an entry that is not an object,
    a ``project`` that is not an object, or a project without a string gid. So a malformed body
    is never read as "no project" and never escapes as a raw ``AttributeError`` or ``TypeError``.
    An empty list, or one of projectless entries only, is a real answer: the task is in no
    project.
    """
    memberships = task.get("memberships")
    if not isinstance(memberships, list):
        raise AsanaResponseError("Asana /tasks/{gid}: memberships is not a list")
    in_a_project = []
    for membership in memberships:
        if not isinstance(membership, dict):
            raise AsanaResponseError("Asana /tasks/{gid}: a membership is not an object")
        project = membership.get("project")
        if project is None:
            continue
        if not isinstance(project, dict) or not isinstance(project.get("gid"), str):
            raise AsanaResponseError("Asana /tasks/{gid}: a membership project has no gid")
        in_a_project.append(membership)
    return in_a_project


def _custom_fields(task: dict) -> list[dict]:
    """A task's ``custom_fields``, checked: a list of objects with a text or null name and value.

    A missing or null list is no fields. Anything else of the wrong shape raises
    ``AsanaResponseError``.
    """
    fields = task.get("custom_fields")
    if fields is None:
        return []
    if not isinstance(fields, list) or not all(
        isinstance(field, dict)
        and all(field.get(key) is None or isinstance(field[key], str) for key in _FIELD_TEXTS)
        for field in fields
    ):
        raise AsanaResponseError("Asana /tasks/{gid}: custom_fields is not a list of fields")
    return fields


def _selected_gids(task: dict, selected: Iterable[str]) -> list[str]:
    """The selected projects a task is in, as sorted gids."""
    return sorted(
        {membership["project"]["gid"] for membership in _memberships(task)} & set(selected)
    )


def _completed_on(task: dict) -> str:
    """The UTC day of ``completed_at`` (``2026-10-09``); empty when it is missing or malformed.

    Never the time: a reopened and completed task would otherwise change its text on every
    completion. A stamp without an offset has no UTC day, so it counts as malformed.
    """
    try:
        stamp = _instant(task["completed_at"])
        return stamp.astimezone(timezone.utc).date().isoformat() if stamp.tzinfo else ""
    except (KeyError, AttributeError, ValueError, OverflowError):
        return ""


def _membership_lines(memberships: list[dict], selected: Iterable[str]) -> list[str]:
    """One ``Project: X (section: Y)`` line per selected project the task is in, sorted.

    Only selected projects are named: a project outside the selection is never synced, so
    its rename would never be seen.
    """
    chosen = set(selected)
    lines = []
    for membership in memberships:
        project = _ref(membership.get("project"))
        name = _name(project)
        if project.get("gid") not in chosen or not name:
            continue
        section = _name(membership.get("section"))
        lines.append(f"Project: {name} (section: {section})" if section else f"Project: {name}")
    return sorted(lines)


def _subtask_lines(subtask: dict) -> list[str]:
    """A checkbox line for a subtask, then indented details that are set."""
    lines = [f"- [{'x' if subtask.get('completed') else ' '}] {_text(subtask.get('name'))}"]
    details = []
    if _name(subtask.get("assignee")):
        details.append(f"Assignee: {_name(subtask['assignee'])}")
    if subtask.get("due_on"):
        details.append(f"Due: {subtask['due_on']}")
    if _text(subtask.get("notes")):
        details.extend(f"Notes: {_text(subtask['notes'])}".splitlines())
    return lines + [f"  {detail}" for detail in details]


def render_task(
    gid: str,
    task: dict,
    comments: list[dict],
    subtasks: list[dict],
    selected_projects: Iterable[str],
) -> dict[str, Any]:
    """A task's row: its fields, notes, subtasks and comments, oldest comment first.

    Nothing volatile is rendered (``modified_at``, counters) and every list is sorted, so an
    unchanged task keeps its content hash and is not extracted again.
    """
    # "Completed", not "Status": workspaces commonly have a custom field named Status.
    lines = [f"Completed: {'yes' if task.get('completed') else 'no'}"]
    if task.get("completed") and (completed_on := _completed_on(task)):
        lines.append(f"Completed on: {completed_on}")
    if _name(task.get("assignee")):
        lines.append(f"Assignee: {_name(task['assignee'])}")
    if task.get("due_on"):
        lines.append(f"Due: {task['due_on']}")
    lines.extend(_membership_lines(_memberships(task), selected_projects))
    fields = sorted(
        (_text(field.get("name")), _text(field.get("display_value")))
        for field in _custom_fields(task)
        if _text(field.get("name")) and _text(field.get("display_value"))
    )
    lines.extend(f"{name}: {value}" for name, value in fields)

    sections = ["\n".join(lines)]
    if _text(task.get("notes")):
        sections.append(_text(task["notes"]))
    if subtasks:
        ordered = sorted(subtasks, key=_gid_order)
        sections.append(
            "Subtasks:\n" + "\n".join(line for sub in ordered for line in _subtask_lines(sub))
        )
    if comments:
        ordered = sorted(comments, key=lambda s: (_text(s.get("created_at")), _gid_order(s)))
        sections.append(
            "Comments:\n"
            + "\n".join(
                f"- {_name(story.get('created_by')) or 'Unknown'}: {_text(story.get('text'))}"
                for story in ordered
            )
        )
    return {
        "id": f"task:{gid}",
        "title": _text(task.get("name")) or "Untitled task",
        "content": "\n\n".join(sections),
        "url": _text(task.get("permalink_url")),
        "_deleted": False,
        # Sorted gids only: the list is hashed, so its order and a project's name must not move it.
        NODE_SET_COLUMN: [f"asana:{gid}" for gid in _selected_gids(task, selected_projects)],
    }


def render_project(gid: str, project: dict) -> dict[str, Any]:
    return {
        "id": f"project:{gid}",
        "title": _text(project.get("name")) or "Untitled project",
        "content": _text(project.get("notes")),
        "url": _text(project.get("permalink_url")),
        "_deleted": False,
        NODE_SET_COLUMN: [f"asana:{gid}"],
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Walker:
    """One run over the selected projects, writing progress into ``state`` at the end.

    State (version ``STATE_VERSION``): ``known_ids`` (row ids emitted and current), ``subtasks``
    (subtask gid -> the task document it is folded into), ``multi`` (task gid -> the selected
    projects of a task that is in more than one), ``selection`` (the selected project gids),
    ``retry`` (tasks kept after a failure), ``pending`` (tasks a stopped run did not reach),
    ``listing`` (only while a listing pass is unfinished: its selection, the result of each
    project listed so far and the progress inside the ones cut) and, per project, the ``cursor``,
    the tasks ``seen`` near it, the Events ``sync`` token and the ``tasks`` last listed. Only the
    final commit (or ``commit_listing``) writes it, so a run that raises leaves it as it was.
    """

    def __init__(
        self,
        client: Any,
        state: dict,
        stats: dict[str, int],
        *,
        project_gids: list[str],
        max_requests: int,
        check_active: Callable[[], None] | None,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        self.projects = project_gids
        self.max_requests = max_requests
        self.check_active = check_active
        self.emitted = 0
        self.used = 0  # the requests of this run, counted against ``max_requests``
        self.known: set[str] = set(state.get("known_ids", []))
        self.subtask_parents: dict[str, str] = dict(state.get("subtasks", {}))
        # parent -> its subtasks, so replacing one task's entries is not a scan of them all
        self.children: dict[str, list[str]] = {}
        for child, parent in self.subtask_parents.items():
            self.children.setdefault(parent, []).append(child)
        self.multi: dict[str, list[str]] = dict(state.get("multi", {}))
        self.retry: set[str] = set(state.get("retry", []))
        self.swept: set[str] = set()  # the gid of every listed task
        self.protected: set[str] = set()  # tasks last listed by a project that is unresolved now
        # Tasks of an unresolved project that wait in ``pending`` or ``retry``: they are not
        # listed this pass, so they are carried over instead of dropped.
        self.held_pending: set[str] = set()
        self.held_retry: set[str] = set()
        self.listing: dict[str, dict] = {}  # project gid -> its ``_list_project`` result
        self.partial: dict[str, dict] = {}  # project gid -> the progress inside a cut project
        self.unresolved: dict[str, int] = {}  # project gid -> the 403 or 404 Asana answered
        self.next_projects: dict[str, dict[str, Any]] = {}
        self.project_rows: list[dict[str, Any]] = []  # of the projects this run listed
        self.order: list[str] = []  # tasks to render: fresh first, then the retries
        self.position = 0  # index in ``order`` of the task being rendered
        self.tail_start = 0  # index in ``order`` where the retries begin
        self.rendered: set[str] = set()
        self.tombstoned: set[str] = set()  # the ids this run forgot
        # A task read or a check of a missing id got an answer after the listing. A run without
        # one made no progress past the listing, see ``commit_progress``.
        self.settled = False
        self.kept: set[str] = set()  # tasks that failed this run and are asked again
        # Tasks in a row that answered 5xx or nothing; the retry tail starts its own count.
        self.outage = 0
        self.listed = False

    # -- requests ----------------------------------------------------------
    def _call(self, path: str, params: dict[str, Any] | None = None):
        """One request, counted against ``max_requests`` (the listing's and the reads' alike).

        A run that spends its budget inside the listing keeps the progress it made, so a small
        budget cannot make a run start over at the same place. The client asks ``check_active``
        before each attempt, so a request the budget refuses is not sent and asks nothing.
        """
        if self.used >= self.max_requests:
            self.stats["failed"] += 1  # the request the run could not afford
            raise _CutShort("budget")
        self.used += 1
        # A call the client retries on a 5xx or a network error still counts once.
        self.stats["requests"] += 1
        try:
            return self.client.request("GET", path, params=params, check_active=self.check_active)
        except AsanaRateLimitedError:
            self.stats["failed"] += 1
            raise _CutShort("rate_limit") from None
        except AsanaAuthError:
            # A rejection before any row left this run raises: the token is bad. Project rows go
            # out right after the listing, so one in the listing always raises; later it stops
            # the run cleanly (the token expired mid-run) only if a row already went out, which
            # a run resuming a pass whose projects earlier runs listed has not.
            if self.emitted:
                self.stats["failed"] += 1
                raise _CutShort("auth") from None
            raise

    def _pages(
        self, path: str, params: dict[str, Any], *, offset: str | None = None
    ) -> Iterator[tuple[list[dict], str | None]]:
        """Each page of a paginated collection as ``(items, next_offset)``, from ``offset``.

        ``next_offset`` is ``None`` after the last page. The offsets of the pages this call has
        read are remembered: one that comes back twice raises instead of looping.
        """
        endpoint = _endpoint(path)
        offsets: set[str] = set()
        while True:
            page_params = {**params, "limit": PAGE_SIZE}
            if offset:
                page_params["offset"] = offset
            body = self._call(path, page_params)
            page = _data(body, list, endpoint)
            if not all(isinstance(item, dict) for item in page):
                raise AsanaResponseError(f"Asana {endpoint}: a listed item is not an object")
            next_page = body.get("next_page")
            if next_page is None:
                offset = None
            else:
                offset = next_page.get("offset") if isinstance(next_page, dict) else None
                if not isinstance(offset, str) or not offset:
                    raise AsanaResponseError(f"Asana {endpoint}: next_page carries no offset")
                if offset in offsets:
                    raise AsanaResponseError(
                        f"Asana {endpoint}: a pagination offset came back twice"
                    )
                offsets.add(offset)
            yield page, offset
            if offset is None:
                return

    def _list_all(self, path: str, params: dict[str, Any]) -> list[dict]:
        """Every item of a paginated collection. Never a partial list."""
        return [item for page, _ in self._pages(path, params) for item in page]

    # -- 1. List -----------------------------------------------------------
    def _poll_events(self, project_gid: str, progress: dict) -> None:
        """Read a project's Events feed into ``progress``, one page at a time.

        ``progress["token"]`` is the sync token to ask next, ``events`` the task gids named so
        far and ``renamed`` whether a project or section rename was among them. ``gap`` is set
        when the feed has one: Asana answers 412 (with a fresh token) when no token is sent or
        the stored one is too old. ``polled`` is set when the feed is read. The sync tokens this
        call has seen are remembered: one that comes back twice while the feed says there is
        more raises instead of looping.
        """
        seen: set[str | None] = {progress["token"]}
        while not progress["polled"]:
            token = progress["token"]
            params = {"resource": project_gid, "opt_fields": _EVENT_FIELDS}
            if token:
                params["sync"] = token
            try:
                body = self._call("/events", params)
            except AsanaSyncTokenExpiredError as expired:
                progress.update(gap=True, token=expired.sync_token, polled=True)
                return
            next_token = body.get("sync")
            if not isinstance(next_token, str) or not next_token:
                raise AsanaResponseError("Asana /events: no sync token")
            events = _data(body, list, "/events")
            more = bool(body.get("has_more"))
            if more and next_token in seen:
                raise AsanaResponseError("Asana /events: a sync token came back twice")
            seen.add(next_token)
            progress["events"] = sorted({*progress["events"], *_event_task_gids(events)})
            progress["renamed"] = progress["renamed"] or _renamed(events)
            progress.update(token=next_token, polled=not more)

    def _read_task_pages(self, project_gid: str, progress: dict) -> None:
        """List a project's tasks page by page into ``progress["listed"]`` (gid -> modified_at).

        ``progress["offset"]`` is where the next page starts, saved after every page, so a run
        cut inside a project resumes at the page it stopped at.
        """
        params = {"project": project_gid, "opt_fields": "modified_at"}
        for page, offset in self._pages("/tasks", params, offset=progress["offset"]):
            for task in page:
                gid, when = task.get("gid"), task.get("modified_at")
                if not isinstance(gid, str) or not _GID.fullmatch(gid):
                    raise AsanaResponseError("Asana /tasks: a task without a usable gid")
                try:
                    when = _stamp(_instant(when))
                except (AttributeError, ValueError, OverflowError):
                    raise AsanaResponseError(
                        "Asana /tasks: a task without a usable modified_at"
                    ) from None
                progress["listed"][gid] = when
            progress["offset"] = offset

    def _read_project(self, project_gid: str) -> dict | None:
        """The project, or ``None`` when Asana answers 403 or 404: it is then unresolved.

        The answer is kept in ``self.unresolved``, so a log can tell the two apart; everything
        else treats them alike, because Asana's answer does not say whether the token has no
        access or the project is gone. Only this call does so. A 401 and any other error raise.
        """
        try:
            body = self._call(f"/projects/{project_gid}", {"opt_fields": _PROJECT_FIELDS})
        except AsanaAPIError as exc:
            if exc.status in (403, 404):
                self.unresolved[project_gid] = exc.status
                return None
            raise
        return _data(body, dict, "/projects")

    def _list_project(self, project_gid: str, project_state: dict) -> dict[str, Any] | None:
        """List one project and return what the pass keeps of it: ids and timestamps, no text.

        The result is ``changed`` (the tasks its window or a full read flags), ``events`` (the
        task gids its events name) and ``next`` (the project's next ``cursor``, ``sync`` token,
        ``seen`` and ``tasks``, the gids listed). ``None`` means the project is unresolved. A
        run cut inside the project leaves its progress in ``self.partial``; the project's row
        goes to ``project_rows`` only once it is listed.
        """
        project = self._read_project(project_gid)
        if project is None:
            return None
        progress = self.partial.setdefault(
            project_gid,
            {
                "token": project_state.get("sync"),
                "polled": False,
                "events": [],
                "renamed": False,
                "gap": False,
                "offset": None,
                "listed": {},
            },
        )
        # Events first: the new token then covers what happens during the listing.
        self._poll_events(project_gid, progress)
        self._read_task_pages(project_gid, progress)
        listed: dict[str, str] = progress["listed"]
        cursor: str = project_state.get("cursor", "")
        # No cursor yet, a gap in the feed or a rename: anything may have changed.
        full = progress["gap"] or not cursor or progress["renamed"]
        # The window starts a little before the cursor, and ``seen`` (gid -> modified_at inside
        # the previous window) tells a task already rendered from one that moved.
        start = _overlap_start(cursor)
        seen: dict[str, str] = project_state.get("seen", {})
        newest = max([cursor, *listed.values()])
        window = _overlap_start(newest)
        del self.partial[project_gid]
        self.project_rows.append(render_project(project_gid, project))
        return {
            "changed": sorted(
                gid
                for gid, when in listed.items()
                if full or (when >= start and seen.get(gid) != when)
            ),
            "events": progress["events"],
            "next": {
                "cursor": newest,
                "sync": progress["token"],
                "seen": {gid: when for gid, when in listed.items() if when >= window},
                "tasks": sorted(listed),
            },
        }

    def _list_phase(self) -> None:
        """Read what the run needs before it renders a task.

        The listing is a pass over the selected projects that can span runs. A project listed by
        an earlier run of the same pass is not listed again, and one cut inside resumes at its
        page (``state["listing"]``, see ``commit_listing``), unless the selection changed since:
        then the pass starts over. A run cut inside the listing raises ``_CutShort`` and keeps
        its progress. A project Asana will not show (403, 404) is unresolved: skipped, tried
        again every run, and its stored tasks stay current.
        """
        previous: dict = self.state.get("projects") or {}
        kept: dict = self.state.get("listing") or {}
        resumed = kept.get("selection") == self.projects
        # Copies: a run that raises later must leave the state as it was.
        self.listing = dict(kept["done"]) if resumed else {}
        self.partial = copy.deepcopy(kept["partial"]) if resumed else {}
        for project_gid in self.projects:
            if project_gid in self.listing:
                continue
            entry = self._list_project(project_gid, previous.get(project_gid) or {})
            if entry is None:
                self.stats["unresolved"] += 1
                # Whatever it had listed goes: it starts at its first page when it returns.
                self.partial.pop(project_gid, None)
                continue
            self.listing[project_gid] = entry

        changed: set[str] = set()
        event_gids: set[str] = set()
        for project_gid in self.projects:
            if project_gid in self.unresolved:
                stored = previous.get(project_gid)
                if stored:
                    # Untouched: its cursor, sync token and seen wait for the project to return.
                    self.next_projects[project_gid] = copy.deepcopy(stored)
                    self.protected.update(stored.get("tasks", []))
                continue
            entry = self.listing[project_gid]
            self.swept.update(entry["next"]["tasks"])
            changed.update(entry["changed"])
            event_gids.update(entry["events"])
            self.next_projects[project_gid] = entry["next"]

        swept = self.swept
        # A task new to the corpus is rendered whatever its timestamp says.
        changed.update(gid for gid in swept if f"task:{gid}" not in self.known)
        changed |= _documents_for(event_gids, swept, self.subtask_parents)
        changed |= set(self.state.get("pending", []))
        # A task in two selected projects names both. Deselecting one leaves the task listed
        # under the other, with a stale line, so it is rendered again.
        removed = set(self.state.get("selection") or []) - set(self.projects)
        changed |= {gid for gid, names in self.multi.items() if removed & set(names)}
        # Only listed tasks are rendered, so every emitted id can be tombstoned later. A queued
        # task of an unresolved project is not listed now, and is held for when it returns.
        queued_pending = set(self.state.get("pending", []))
        self.held_pending = (queued_pending & self.protected) - swept
        self.held_retry = (self.retry & self.protected) - swept
        self.retry &= swept
        fresh = sorted((changed & swept) - self.retry)
        self.order = fresh + sorted(self.retry)
        self.tail_start = len(fresh)
        self.stats["scanned"] = len(swept)
        self.stats["skipped"] = len(swept) - len(self.order)
        self.listed = True

    def _still_selected(self, row_id: str) -> bool:
        """Whether a known id the listing lacks is kept: a task that still has a selected project.

        A listing can miss a live task: it moved between projects while the pass ran or between
        the runs that made it, or a page shifted on resume. So a missing task is asked for its
        memberships (one request, counted in ``max_requests``): a 403 or 404, or no selected
        project, means it is gone and the caller forgets it; a selected project keeps it known
        (its project's events or listing render it on a later pass). An answer of 5xx or nothing
        is unknown, not gone: the task stays known and counts in ``failed`` like a read that
        failed, and toward the outage streak, so a hung Asana does not spend the budget on
        checks either. Any other error raises, a 200 of the wrong shape (or whose ``memberships``
        is malformed, see ``_memberships``; entries without a project are skipped) raises
        ``AsanaResponseError``, and a 429 or a spent budget raises ``_CutShort``. This differs
        from a task read, which keeps its task for a retry on such a body: the check decides
        whether a task is forgotten, so a body it cannot read raises instead of being guessed at.
        """
        if not row_id.startswith("task:"):
            return False
        path = f"/tasks/{row_id.removeprefix('task:')}"
        try:
            body = self._call(path, {"opt_fields": "memberships.project.gid"})
        except AsanaAPIError as exc:
            if exc.status in (403, 404):
                return False
            if exc.status == 0 or exc.status >= 500:
                self.stats["failed"] += 1
                self._count_outage()
                return True
            raise
        self.outage = 0
        return bool(_selected_gids(_data(body, dict, "/tasks/{gid}"), self.projects))

    def _count_outage(self) -> None:
        """Count an answer of 5xx or nothing toward the outage streak; three in a row stop the run."""
        self.outage += 1
        if self.outage >= _OUTAGE_STREAK:
            raise _CutShort("unavailable")

    def _current_ids(self) -> set[str]:
        tasks = self.swept | self.protected
        return {f"task:{gid}" for gid in tasks} | {f"project:{g}" for g in self.projects}

    # -- 2. Render ---------------------------------------------------------
    def _fetch_task(self, gid: str) -> tuple[dict, list[dict], list[dict]]:
        """``(task, comments, subtasks)`` of one task.

        Raises ``AsanaAPIError`` for a failing answer and ``AsanaResponseError`` for a 200 of the
        wrong shape (the task, a comment or a subtask that is not an object, a bad pagination);
        ``rows`` keeps the task for a retry on either.
        """
        path = f"/tasks/{gid}"
        task = _data(self._call(path, {"opt_fields": _TASK_FIELDS}), dict, "/tasks/{gid}")
        stories = self._list_all(f"{path}/stories", {"opt_fields": _STORY_FIELDS})
        subtasks = self._list_all(f"{path}/subtasks", {"opt_fields": _SUBTASK_FIELDS})
        # The stories feed also holds system entries ("added to project", ...).
        comments = [story for story in stories if story.get("resource_subtype") == "comment_added"]
        return task, comments, subtasks

    def _failed(self, gid: str, exc: AsanaSourceError) -> None:
        """Keep a task that could not be read, and stop if Asana as a whole is failing.

        The task is still listed, so it is not gone: its old document stays and it is asked
        again next run, after every fresh task. Only answers of 5xx or nothing count toward an
        outage, in the fresh tasks and, with a count of their own, in the retry tail: a task that
        keeps answering 403 or 404 never stops the others.
        """
        self.stats["failed"] += 1
        self.kept.add(gid)
        # The status and the endpoint, never Asana's explanation and never the task's gid.
        logger.debug("Asana: a task was kept for a retry next run (%s).", exc)
        if isinstance(exc, AsanaAPIError) and (exc.status == 0 or exc.status >= 500):
            self._count_outage()

    def _remember(self, gid: str, task: dict, subtasks: list[dict]) -> None:
        """Replace this task's subtask entries and note whether it is in several projects."""
        for child in self.children.pop(gid, []):
            if self.subtask_parents.get(child) == gid:
                del self.subtask_parents[child]
        children = [sub["gid"] for sub in subtasks if isinstance(sub.get("gid"), str)]
        for child in children:
            self.subtask_parents[child] = gid
        self.children[gid] = children
        selected = _selected_gids(task, self.projects)
        if len(selected) > 1:
            self.multi[gid] = selected
        else:
            self.multi.pop(gid, None)

    def _project_row_ids(self) -> list[str]:
        return [row["id"] for row in self.project_rows]

    def _project_rows(self) -> Iterator[dict[str, Any]]:
        for row in self.project_rows:
            self.emitted += 1
            yield row

    def rows(self) -> Iterator[dict[str, Any]]:
        try:
            self._list_phase()
        except _CutShort:
            # The projects listed so far go out now: their rows are whole documents.
            yield from self._project_rows()
            raise
        yield from self._project_rows()
        for index, gid in enumerate(self.order):
            self.position = index
            if index == self.tail_start:
                self.outage = 0
            try:
                task, comments, subtasks = self._fetch_task(gid)
            except (AsanaAPIError, AsanaResponseError) as exc:
                self._failed(gid, exc)
                continue
            self.outage = 0
            self.settled = True
            try:
                selected = _selected_gids(task, self.projects)
                row = (
                    render_task(gid, task, comments, subtasks, self.projects) if selected else None
                )
            except AsanaResponseError as exc:
                # A body of the wrong shape is a task Asana did not answer properly, like the
                # read above: kept for a retry, never skipped silently, never raising.
                self._failed(gid, exc)
                continue
            if row is None:
                # It left the selection since the listing: no document is worth building. It
                # stays known, and the next pass finds it missing and forgets it.
                continue
            self._remember(gid, task, subtasks)
            self.rendered.add(gid)
            self.emitted += 1
            yield row

        # The listing is complete: only now may a row that is no longer listed be forgotten.
        # Each missing id is settled on its own, so a check cut short keeps the tombstones
        # already out and the rest stay known for the next pass.
        self.position = len(self.order)
        self.outage = 0  # the checks count their own streak
        for row_id in sorted(self.known - self._current_ids()):
            verdict = self._still_selected(row_id)
            self.settled = True
            if verdict:
                continue
            self.tombstoned.add(row_id)
            self.stats["deleted"] += 1
            yield {"id": row_id, "_deleted": True}
        self._commit(self.known - self.tombstoned | self._emitted_ids(), [], self.kept)

    # -- 3. Commit ---------------------------------------------------------
    def _emitted_ids(self) -> set[str]:
        projects = set(self.projects) - set(self.unresolved)
        return {f"task:{gid}" for gid in self.rendered} | {f"project:{g}" for g in projects}

    def commit_progress(self) -> bool:
        """After a clean stop: keep what was rendered, queue what was not, delete nothing.

        Returns whether the state changed: a stop that kept only a listing that is already kept
        changed nothing, and the next run would stop at the same place.

        The cursors and sync tokens advance, because every task they covered is either rendered
        or queued. A stop inside the listing, or before any task read or check got an answer,
        only keeps the listing, see ``commit_listing``.
        """
        if not (self.settled or self.kept):
            # Nothing past the listing got an answer (or the listing itself was cut): the pass
            # keeps its listing, complete or not, and the next run starts from it. Committing
            # instead would list again first, and a budget or rate limit that the listing alone
            # uses up would never leave room for a task or a check.
            return self.commit_listing()
        rest = self.order[self.position :]
        retry = self.kept | (self.retry & set(rest))
        pending = [gid for gid in rest if gid not in retry]
        self._commit(self.known - self.tombstoned | self._emitted_ids(), pending, retry)
        return True

    def commit_listing(self) -> bool:
        """After a stop before the pass is rendered: keep its listing, move nothing else.

        ``state["listing"]`` holds the selection of the pass, per finished project the ids and
        timestamps ``_list_project`` returned, and per project cut inside its progress so far
        (events, task gids with their timestamps, page offset): never text. The ids can exceed
        ``known_ids`` until the pass completes. The next run of the same selection lists only
        what is left. The project rows of this run are out, so their ids join ``known_ids`` (a
        later deselection then forgets them). Nothing is rendered, forgotten or advanced
        (``selection``, the cursors and the sync tokens stay as the last complete pass left
        them), because a pass that has not listed every project knows neither which task left
        nor what is left to render. ``_commit`` clears it when the pass completes. A run that
        added nothing to what the state holds leaves it as it is, and returns ``False``.
        """
        kept: dict = self.state.get("listing") or {}
        if self.listing == kept.get("done", {}) and self.partial == kept.get("partial", {}):
            return False
        self.state["v"] = STATE_VERSION
        self.state["known_ids"] = sorted(self.known | set(self._project_row_ids()))
        self.state["listing"] = {
            "selection": list(self.projects),
            "done": self.listing,
            "partial": self.partial,
        }
        return True

    def _commit(self, known: set[str], pending: list[str], retry: set[str]) -> None:
        retry = retry | self.held_retry
        pending = pending + sorted(self.held_pending - set(pending))
        unsettled = retry | set(pending)
        for project_gid, project_state in self.next_projects.items():
            if project_gid in self.unresolved:
                continue
            # A task not rendered must not look settled inside the overlap window.
            project_state["seen"] = {
                gid: when for gid, when in project_state["seen"].items() if gid not in unsettled
            }
        self.state["v"] = STATE_VERSION
        self.state["selection"] = list(self.projects)
        self.state["known_ids"] = sorted(known)
        self.state["subtasks"] = {
            child: parent
            for child, parent in sorted(self.subtask_parents.items())
            if f"task:{parent}" in known
        }
        self.state["multi"] = {
            gid: names for gid, names in sorted(self.multi.items()) if f"task:{gid}" in known
        }
        self.state["projects"] = self.next_projects
        self.state.pop("listing", None)
        self.state["retry"] = sorted(retry)
        self.state["pending"] = pending
        # This run's own counts: a run that raises, or stops before it lists, commits nothing and
        # reports none, never what an earlier run left in the state.
        self.stats["kept"] = len(retry)
        self.stats["pending"] = len(pending)


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them on any exit.

    ``unresolved`` counts the projects Asana would not show this run. A reason key
    (``failed_rate_limit``, ``failed_budget``, ``failed_unavailable``, ``failed_auth``) joins them
    only when a run stops for it.
    """
    return dict.fromkeys(
        ("scanned", "skipped", "deleted", "failed", "kept", "pending", "requests", "unresolved"), 0
    )


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    project_gids: list[str],
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield the changed rows of the selected projects. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        state,
        stats,
        project_gids=project_gids,
        max_requests=max_requests,
        check_active=check_active,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        changed = walker.commit_progress()
        stats[f"failed_{cut.reason}"] = 1
        logger.warning("Asana sync stopped early (%s); it resumes on the next run.", cut.reason)
        if cut.reason == "budget" and not changed:
            # Nothing was kept, so the next run starts at the same place and spends the same
            # budget the same way: the budget cannot cover this selection. A 429 is not this,
            # it is the provider's own limit and clears with time.
            raise AsanaSourceError(
                f"Asana: max_requests={max_requests} is too small for this selection. The run "
                f"spent {walker.used} requests, {len(walker.unresolved)} of them re-reading "
                "projects Asana would not show, and the budget ran out before one task read "
                f"({_TASK_REQUESTS} requests) or one check of a missing task (1 request) could "
                "answer, so the next run would stop at the same place. Raise max_requests."
            ) from None
    except AsanaSourceError as exc:
        # The run raises: dlt drops its state, not this object, and a host reads these counts
        # after the raise.
        stats["failed"] += 1
        if isinstance(exc, AsanaAuthError):
            stats["failed_auth"] = 1
        raise
    finally:
        # Counts only, no gids: a token's reach is not for a log.
        forbidden = sum(status == 403 for status in walker.unresolved.values())
        if forbidden:
            logger.warning(
                "Asana: %d project(s) answered 403, the token may not reach them; they are "
                "skipped and asked again next run.",
                forbidden,
            )
        if len(walker.unresolved) > forbidden:
            logger.warning(
                "Asana: %d project(s) answered 404, they are deleted or the gid is wrong; they "
                "are skipped and asked again next run.",
                len(walker.unresolved) - forbidden,
            )
        if walker.kept:
            # A task Asana will not show is an expected state, so one line a run, with no ids.
            logger.warning(
                "Asana: %d task(s) could not be read; their old documents stay and they are "
                "asked again next run.",
                len(walker.kept),
            )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def _check_resource_name(name: Any) -> None:
    """A non-blank string that is not a name dlt keeps for itself.

    ``_dlt_loads``, ``_dlt_pipeline_state`` and ``_dlt_version`` are dlt's own tables: a
    resource of that name fails the load after the walk has spent its requests.
    """
    from dlt.common.normalizers.naming.snake_case import NamingConvention

    if not isinstance(name, str) or not name.strip():
        raise ValueError("asana_source needs a resource_name, one per Asana connection")
    if NamingConvention().normalize_table_identifier(name).startswith("_dlt_"):
        raise ValueError("resource_name must not start with _dlt_, which dlt keeps for itself")


def _project_gid_list(project_gids: list[str] | tuple[str, ...]) -> list[str]:
    """``project_gids`` sorted and without repeats.

    Anything but a non-empty list or tuple of gid strings is refused. A bare string would be read
    as one project per letter, and an empty selection would forget every synced task.
    """
    if (
        not isinstance(project_gids, (list, tuple))
        or not project_gids
        or not all(isinstance(gid, str) and _GID.fullmatch(gid) for gid in project_gids)
    ):
        raise ValueError(
            'project_gids must be a non-empty list or tuple of project gid strings, like ["<gid>"].'
        )
    return sorted(set(project_gids))


def asana_source(
    *,
    resource_name: str,
    project_gids: list[str] | tuple[str, ...],
    access_token: str | None = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields the selected Asana projects' documents.

    Args:
        resource_name: Stable name of this connection. It names the staging table and scopes the
            sync state, so two connections in one dataset must never share one. There is no
            default.
        project_gids: The projects to sync, as gid strings. Required: an empty selection is an
            error, never "everything". A workspace-wide sync is a list of its project gids.
            Tasks that leave the selection are forgotten.
        access_token: Asana personal access token or OAuth access token. Used to build the
            client when ``service`` is omitted. Never read from the environment.
        service: Pre-built client (see :func:`build_asana_service`). Mainly an injection point
            for tests and hosts. It must accept ``request(method, path, *, params,
            check_active)`` and call ``check_active`` before each attempt it makes.
        check_active: Optional host authorization checkpoint, called before each attempt of each
            request, a retry included. What it raises stops the run as it is.
        max_requests: Budget of the requests of one run, the listing's and the task reads' (a
            call the client retries counts once), so at least 3 (a task costs three). A run cut
            by it inside the listing keeps its progress and the next run continues there;
            rendering waits for a complete listing.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``, ``write_disposition="merge"``
        and a ``_deleted`` hard-delete column.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 2:
        raise RuntimeError(
            "Asana sync requires a Cognee build that reads per-row node sets "
            "(DOCUMENT_SYNC_VERSION >= 2). Upgrade Cognee before syncing."
        )
    _check_resource_name(resource_name)
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
        raise ValueError("max_requests must be a positive int")
    if max_requests < _TASK_REQUESTS:
        raise ValueError(
            f"max_requests must be at least {_TASK_REQUESTS}, the requests that read one task"
        )
    projects = _project_gid_list(project_gids)
    if service is None and not access_token:
        raise ValueError("asana_source needs a service or an access_token")

    # Built here so a malformed token fails at construction and the closure holds no raw token.
    client = service if service is not None else AsanaClient(access_token or "")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def asana_documents():
        stats.clear()
        stats.update(new_stats())
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            project_gids=projects,
            max_requests=max_requests,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = asana_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "asana")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
