"""Todoist connector for cognee: a ``dlt`` source that turns an account into memory.

One source is one Todoist API token. It yields one document per task (with its comments
folded in) and one per project as flat rows, and is meant to be handed directly to
:func:`cognee.remember`::

    import cognee
    from cognee_connector_todoist import todoist_source

    await cognee.remember(
        todoist_source(
            resource_name="todoist_me",
            access_token="<API token>",
        ),
        dataset_name="todoist",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are
  prefixed by kind (``task:<id>``, ``project:<id>``). Every column feeds the stored
  document identity, so the rendering is frozen and deterministic: no timestamps other
  than the day a task was completed, every collection sorted. A tombstone carries no
  node set.
* **Node set** ``todoist:<project id>`` per task (the project the task is in) and
  ``todoist:<its own id>`` for a project row, so a ``recall`` can be scoped to one
  project. Only ids are used: renaming a project changes no node set. It is hashed
  into the document identity, so a task that moves to another project is extracted
  once more.
* **Selection** is ``project_ids``, optional: ``None`` syncs every project the token
  sees, a list limits the sync to those projects. Changing the selection restarts the
  cursor: the next run sends ``"*"`` again and treats the live answer as the whole
  corpus, which both backfills a newly selected project (its changes lie behind the old
  token) and forgets the tasks and projects that left the selection.
* **Incremental** in one cursor: the Sync API's ``sync_token``. The first run sends
  ``"*"`` and gets a full copy; every later run sends the stored token and gets exactly
  what changed — new or edited tasks, projects and comments, and deletions, which the
  feed marks with ``is_deleted``. The token is stored only after the whole answer was
  processed, so a run that fails or stops re-fetches the same changes next time.
* **Comments** are not rows of their own: they are folded into their task's document.
  The Sync feed carries a changed comment without the task's other comments, so a task
  is re-rendered whenever one of its comments changes, and its comments are always read
  in full through the REST API.
* **Forget-on-delete.** The Sync feed is authoritative about deletions: a task or
  project deleted in Todoist arrives with ``is_deleted`` and is tombstoned at once, and
  a deleted project also tombstones the tasks that belonged to it. A run tombstones
  only ids this connection knows (its own state), and only after the sync call that
  reported the deletion succeeded; a run that raises before that forgets nothing.
* **A run is bounded.** ``max_requests`` bounds the requests of one run (the sync call,
  one per task whose body the feed did not carry, one per rendered task's comments). A
  run the budget cuts stops cleanly, keeps no state and advances no cursor, so the next
  run re-fetches the same changes and repeats only idempotent upserts; it never sleeps
  on a 429: the client retries a 429 a bounded number of times and then raises, dlt
  rolls the run back, and the changes wait for the next run.
* **Token** is a plain API token. It lives in the client object only; errors carry
  statuses and endpoints, never the token, a task id or Todoist's own explanation.

Privacy
-------
This connector reads the names, descriptions and comments of the tasks of the account,
and the project names. It is **opt-in**: nothing is fetched until you construct a source
and call ``remember``. What it reads goes to your configured LLM and embedding
providers, and anyone with read access to the target dataset can see it. Email addresses
are never requested, but people type them into comments and descriptions.
"""

import logging
import re
from collections.abc import Callable, Iterator
from typing import Any

from cognee_connector_todoist.client import (
    TodoistAPIError,
    TodoistAuthError,
    TodoistClient,
    TodoistRateLimitedError,
    TodoistResponseError,
)
from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = logging.getLogger(__name__)

# Todoist's largest page of the REST endpoints used here.
PAGE_SIZE = 100
# Budget of the requests of one run. Our own choice, not an API limit: Todoist documents
# only rate limits per minute. A task whose body the feed did not carry costs two
# requests (the task and its comments), a changed task one (its comments).
DEFAULT_MAX_REQUESTS = 1000
# Version of the resource state; a later release reads it to migrate an older layout.
STATE_VERSION = 1
# The kinds the Sync API call asks for. "items" are tasks, "notes" their comments.
RESOURCE_TYPES = ["items", "projects", "notes"]

_EXTRA_HINT = "Set access_token= or build the source with service=build_todoist_service(token)."

# A project or task id goes into a URL, so no character that can change the path may get
# through. Todoist's ids are digits.
_ID = re.compile(r"[0-9A-Za-z_-]+")

_PRIORITY = {1: "normal", 2: "medium", 3: "high", 4: "urgent"}

TASK_APP_URL = "https://app.todoist.com/app/task/{id}"


# ---------------------------------------------------------------------------
# Rendering: deterministic, no volatile fields.
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _utc_day(stamp: Any) -> str:
    """The UTC day of an ISO timestamp (``2026-10-14``); empty when it has none."""
    if not isinstance(stamp, str) or "T" not in stamp:
        return ""
    return stamp.split("T", 1)[0]


def render_task(item_id: str, item: dict, comments: list[dict]) -> dict[str, Any]:
    """A task's row: its fields, description and comments, oldest comment first.

    Nothing volatile is rendered (``updated_at``, counters) and every list is sorted, so
    an unchanged task keeps its content hash and is not extracted again.
    """
    lines = [f"Completed: {'yes' if item.get('checked') else 'no'}"]
    if item.get("checked") and (day := _utc_day(item.get("completed_at"))):
        lines.append(f"Completed on: {day}")
    priority = item.get("priority")
    if isinstance(priority, int) and not isinstance(priority, bool):
        lines.append(f"Priority: {_PRIORITY.get(priority, str(priority))}")
    due = item.get("due")
    if isinstance(due, dict):
        due_text = _text(due.get("date")) or _text(due.get("string"))
        if due_text:
            lines.append(f"Due: {due_text}")
    labels = sorted({_text(label) for label in item.get("labels") or []} - {""})
    if labels:
        lines.append("Labels: " + ", ".join(labels))

    sections = ["\n".join(lines)]
    if description := _text(item.get("description")):
        sections.append(description)
    if comments:
        ordered = sorted(
            comments,
            key=lambda note: (_text(note.get("posted_at")), _text(note.get("id"))),
        )
        sections.append(
            "Comments:\n" + "\n".join(f"- {_text(note.get('content'))}" for note in ordered)
        )
    project_id = _text(item.get("project_id"))
    return {
        "id": f"task:{item_id}",
        "title": _text(item.get("content")) or "Untitled task",
        "content": "\n\n".join(sections),
        "url": TASK_APP_URL.format(id=item_id),
        "_deleted": False,
        NODE_SET_COLUMN: [f"todoist:{project_id}"] if project_id else [],
    }


def render_project(project_id: str, project: dict) -> dict[str, Any]:
    return {
        "id": f"project:{project_id}",
        "title": _text(project.get("name")) or "Untitled project",
        "content": "",
        "url": "",
        "_deleted": False,
        NODE_SET_COLUMN: [f"todoist:{project_id}"],
    }


# ---------------------------------------------------------------------------
# The Sync API answer, checked.
# ---------------------------------------------------------------------------
def _checked_object_list(answer: dict, key: str) -> list[dict]:
    """``answer[key]``, which must be a list of objects. Never read as an empty feed."""
    value = answer.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TodoistResponseError(f"Todoist /sync/v9/sync: {key} is not a list of objects")
    return value


def _object_id(obj: dict, what: str) -> str:
    """The id of a sync object, which must be a non-empty string."""
    object_id = obj.get("id")
    if not isinstance(object_id, str) or not object_id:
        raise TodoistResponseError(f"Todoist /sync/v9/sync: {what} without a usable id")
    return object_id


def _is_deleted(obj: dict) -> bool:
    return bool(obj.get("is_deleted"))


def _note_parent_id(note: dict) -> str:
    """The id of the task a note belongs to: ``item_id``, or legacy ``task_id``."""
    for key in ("item_id", "task_id"):
        parent = note.get(key)
        if isinstance(parent, str) and parent:
            return parent
    raise TodoistResponseError("Todoist /sync/v9/sync: a note without a usable item_id")


def _split_answer(answer: dict) -> dict[str, Any]:
    """The sync answer into the six buckets the run acts on, every id checked."""
    items = _checked_object_list(answer, "items")
    projects = _checked_object_list(answer, "projects")
    notes = _checked_object_list(answer, "notes")
    return {
        "changed_tasks": {_object_id(i, "an item"): i for i in items if not _is_deleted(i)},
        "deleted_tasks": sorted(_object_id(i, "an item") for i in items if _is_deleted(i)),
        "changed_projects": {_object_id(p, "a project"): p for p in projects if not _is_deleted(p)},
        "deleted_projects": sorted(_object_id(p, "a project") for p in projects if _is_deleted(p)),
        # A changed or deleted note re-renders its task: the feed carries only the note.
        "comment_parents": sorted({_note_parent_id(n) for n in notes}),
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them on any exit."""
    return dict.fromkeys(("changes", "rendered", "deleted", "failed", "requests"), 0)


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    project_ids: list[str] | None,
    max_requests: int,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield the changed rows of one sync. Pure of dlt, so tests drive it with a dict.

    A run that raises (an API error out of retries, a malformed answer) emits tombstones
    for nothing: tombstones are computed only after the one sync call that reports the
    deletions has succeeded, and a render failure aborts the run before the tombstone
    phase. dlt drops the state of a raised run, so the next one re-fetches and forgets
    nothing a failed run seemed to know.
    """
    selected = set(project_ids) if project_ids is not None else None
    used = 0

    # Full mode answers the whole account with the "*" cursor. It runs on the first sync
    # and whenever the selection changed: an incremental cursor cannot backfill a newly
    # selected project (its changes are already behind the token), and cannot see what
    # left the selection. In full mode the live answer is the whole corpus, so anything
    # known that is not live and selected is tombstoned.
    recorded = state.get("selection")
    recorded = sorted(recorded) if isinstance(recorded, list) else None
    full_mode = "sync_token" not in state or recorded != (
        sorted(selected) if selected is not None else None
    )
    token_in = "*" if full_mode else state["sync_token"]

    def spend(estimate: int) -> None:
        """Refuse a request the run cannot afford: the run stops cleanly instead."""
        nonlocal used
        if used + estimate > max_requests:
            stats["failed"] += 1
            raise _CutShort("budget")
        used += estimate

    spend(1)
    stats["requests"] += 1
    answer = client.sync(token_in, RESOURCE_TYPES, check_active=check_active)
    if not isinstance(answer, dict) or not isinstance(answer.get("sync_token"), str):
        raise TodoistResponseError("Todoist /sync/v9/sync: no sync_token in the answer")
    split = _split_answer(answer)
    stats["changes"] = (
        len(answer.get("items", []))
        + len(answer.get("projects", []))
        + len(answer.get("notes", []))
    )

    known_tasks: dict[str, str] = dict(state.get("tasks", {}))  # task id -> project id
    known_projects: set[str] = set(state.get("projects", []))
    tombstoned: set[str] = set()
    rendered_tasks: dict[str, str] = {}  # task id -> project id, this run
    rendered_projects: set[str] = set()

    def in_selection(project_id: str) -> bool:
        return selected is None or project_id in selected

    # -- 1. Render the changed projects and tasks (upserts first, tombstones last) ----
    project_rows = sorted(pid for pid in split["changed_projects"] if in_selection(pid))
    for project_id in project_rows:
        yield render_project(project_id, split["changed_projects"][project_id])
        rendered_projects.add(project_id)

    if full_mode:
        to_render = sorted(split["changed_tasks"])
    else:
        to_render = sorted(
            (set(split["changed_tasks"]) | set(split["comment_parents"]))
            - set(split["deleted_tasks"])
        )
    for task_id in to_render:
        if check_active is not None:
            check_active()
        item = split["changed_tasks"].get(task_id)
        if item is not None and not in_selection(_text(item.get("project_id"))):
            # Outside the selection and the feed already told us: no request is spent.
            continue
        if item is None:
            # Only a comment changed: the feed carries the note, not the task's body.
            spend(2)
            stats["requests"] += 2
            try:
                item = client.get_task(task_id, check_active=check_active)
            except TodoistAPIError as exc:
                if exc.status in (403, 404):
                    logger.warning("Todoist: task %s is gone, skipping it: %s", task_id, exc)
                    continue
                raise
            if not isinstance(item, dict):
                raise TodoistResponseError("Todoist /tasks/{id}: HTTP 200 without a JSON object")
        else:
            spend(1)
            stats["requests"] += 1
        project_id = _text(item.get("project_id"))
        if not in_selection(project_id):
            # Fetched because only its comment changed: outside the selection after all.
            continue
        comments = client.list_comments(task_id, check_active=check_active)
        row = render_task(task_id, item, comments)
        rendered_tasks[task_id] = project_id
        stats["rendered"] += 1
        yield row

    # -- 2. Tombstones ------------------------------------------------------------
    def tombstone(row_id: str) -> dict:
        tombstoned.add(row_id)
        stats["deleted"] += 1
        return {"id": row_id, "_deleted": True}

    if full_mode:
        # The live answer is the corpus: anything known that is neither live nor
        # selected is gone — deleted in Todoist or moved out of the selection.
        for task_id in sorted(set(known_tasks) - set(rendered_tasks)):
            yield tombstone(f"task:{task_id}")
        for project_id in sorted(known_projects - rendered_projects):
            yield tombstone(f"project:{project_id}")
    else:
        for task_id in split["deleted_tasks"]:
            if task_id in known_tasks:
                yield tombstone(f"task:{task_id}")
        for project_id in split["deleted_projects"]:
            if project_id in known_projects:
                yield tombstone(f"project:{project_id}")
            for task_id, task_project in sorted(known_tasks.items()):
                if task_project == project_id and f"task:{task_id}" not in tombstoned:
                    yield tombstone(f"task:{task_id}")
        if selected is not None:
            for task_id, task_project in sorted(known_tasks.items()):
                if task_project not in selected and f"task:{task_id}" not in tombstoned:
                    yield tombstone(f"task:{task_id}")
            for project_id in sorted(known_projects - selected):
                if f"project:{project_id}" not in tombstoned:
                    yield tombstone(f"project:{project_id}")

    # -- 3. Commit: only a run that processed the whole answer advances the cursor ----
    state["v"] = STATE_VERSION
    state["sync_token"] = answer["sync_token"]
    state["selection"] = sorted(selected) if selected is not None else None
    state["tasks"] = {
        **{tid: pid for tid, pid in known_tasks.items() if f"task:{tid}" not in tombstoned},
        **rendered_tasks,
    }
    state["projects"] = (
        sorted(rendered_projects)
        if full_mode
        else sorted(
            (known_projects - set(split["deleted_projects"]) - (selected or set()))
            | set(split["changed_projects"])
        )
    )
    logger.info(
        "Todoist: %d change(s), %d rendered, %d deletion(s), %d request(s).",
        stats["changes"],
        stats["rendered"],
        stats["deleted"],
        stats["requests"],
    )


def _run_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    project_ids: list[str] | None,
    max_requests: int,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """``_iter_rows`` with the run-level failure mapping (reason keys, clean stops)."""
    try:
        yield from dlt_utils.guarded_rows(
            _iter_rows(
                client,
                state,
                stats,
                project_ids=project_ids,
                max_requests=max_requests,
                check_active=check_active,
            ),
            check_active,
        )
    except _CutShort as cut:
        stats[f"failed_{cut.reason}"] = 1
        logger.warning(
            "Todoist sync stopped early (%s); the cursor was not advanced, so the next "
            "run re-fetches the same changes.",
            cut.reason,
        )
    except TodoistAuthError:
        stats["failed"] += 1
        stats["failed_auth"] = 1
        raise
    except TodoistRateLimitedError:
        stats["failed"] += 1
        stats["failed_rate_limit"] = 1
        raise
    except TodoistAPIError:
        stats["failed"] += 1
        raise
    except TodoistResponseError:
        stats["failed"] += 1
        raise


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
        raise ValueError("todoist_source needs a resource_name, one per Todoist connection")
    if NamingConvention().normalize_table_identifier(name).startswith("_dlt_"):
        raise ValueError("resource_name must not start with _dlt_, which dlt keeps for itself")


def _project_id_list(project_ids: list[str] | tuple[str, ...] | None) -> list[str] | None:
    """``project_ids`` sorted and without repeats; ``None`` stays ``None`` (every project).

    Anything but a list or tuple of id strings is refused. A bare string would be read as
    one project per letter, and an empty selection would forget every synced task.
    """
    if project_ids is None:
        return None
    if (
        not isinstance(project_ids, (list, tuple))
        or not project_ids
        or not all(isinstance(pid, str) and _ID.fullmatch(pid) for pid in project_ids)
    ):
        raise ValueError(
            "project_ids must be None or a non-empty list or tuple of project id strings, "
            'like ["<id>"].'
        )
    return sorted(set(project_ids))


def todoist_source(
    *,
    resource_name: str,
    access_token: str | None = None,
    service: Any = None,
    project_ids: list[str] | tuple[str, ...] | None = None,
    check_active: Callable[[], None] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields the account's Todoist documents.

    Args:
        resource_name: Stable name of this connection. It names the staging table and
            scopes the sync state, so two connections in one dataset must never share
            one. There is no default.
        access_token: Todoist API token. Used to build the client when ``service`` is
            omitted. Never read from the environment.
        service: Pre-built client (see :func:`build_todoist_service`). Mainly an
            injection point for tests and hosts. It must offer ``sync(sync_token,
            resource_types)``, ``get_task(task_id)`` and ``list_comments(task_id)`` and
            call ``check_active`` before each attempt it makes.
        project_ids: The projects to sync, as id strings. ``None`` (the default) syncs
            every project the token sees; a list limits the sync to those projects, and
            tasks and projects that leave the selection are forgotten on the next run.
        check_active: Optional host authorization checkpoint, called around every row.
            What it raises stops the run as it is.
        max_requests: Budget of the requests of one run: the sync call (1), one per task
            whose body the feed did not carry, one per rendered task's comments. A run
            the budget cuts stops cleanly, advances no cursor and forgets nothing.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``, ``write_disposition="merge"``
        and a ``_deleted`` hard-delete column.
    """
    import dlt

    dlt_utils.require_document_sync_version(
        2, provider="Todoist", feature="reads per-row node sets"
    )
    _check_resource_name(resource_name)
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
        raise ValueError("max_requests must be a positive int")
    selection = _project_id_list(project_ids)
    if service is None and not access_token:
        raise ValueError(f"todoist_source needs a service or an access_token: {_EXTRA_HINT}")

    # Built here so a malformed token fails at construction and the closure holds no raw token.
    client = service if service is not None else TodoistClient(access_token or "")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def todoist_documents():
        stats.clear()
        stats.update(new_stats())
        yield from _run_rows(
            client,
            dlt.current.resource_state(),
            stats,
            project_ids=selection,
            max_requests=max_requests,
            check_active=check_active,
        )

    resource = todoist_documents()
    # Opt into the document ingestion path: each row (id/title/content/url, plus
    # cognee_node_set) becomes a text document that flows through normal cognify.
    # resolve_dlt_sources reads this marker; it never imports this connector.
    setattr(resource, DOCUMENT_SOURCE_ATTR, "todoist")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    # Host-readable diagnostics contain counts only, never task titles or content.
    resource.cognee_sync_stats = stats
    return resource


def build_todoist_service(access_token: str) -> TodoistClient:
    """Wrap a host-owned API token in a client (an alias kept next to the source)."""
    return TodoistClient(access_token)
