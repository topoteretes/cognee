# Adapted from topoteretes/cognee-community (Apache-2.0), PR #349, commit ef32699 (AbdulWasih05).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""An Asana task is re-extracted only when its rendered text changes, and nothing but a complete
listing ever forgets one.

The walk is driven with a plain dict as dlt resource state, and then through a real ``dlt``
pipeline into SQLite. What a task that cannot be read does is the heart of it: while it is still
listed it keeps its old document, is asked again on the next run after every fresh task, never
counts toward an outage and never stops another task. So does a 200 of the wrong shape in its own
read, while one in a listing, the events feed, a project read or a check raises. A run that stops
(a rejected token after rows, a rate limit, a spent budget, an outage) keeps what it emitted,
queues the rest and forgets nothing; a 429 is never waited out.
"""

import copy
import json
import random
import re

import pytest
from fake_asana import FakeAsana, Resp, serve_over_http

from cognee.tasks.ingestion.connectors import asana as asana_module
from cognee.tasks.ingestion.connectors import asana_source
from cognee.tasks.ingestion.connectors.asana import (
    STATE_VERSION,
    AsanaAPIError,
    AsanaAuthError,
    AsanaClient,
    AsanaResponseError,
    AsanaSourceError,
    _documents_for,
    _event_task_gids,
    _instant,
    _iter_rows,
    _overlap_start,
    _renamed,
    _stamp,
    new_stats,
    render_task,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_COLUMN_HINTS,
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)
from cognee.tasks.ingestion.ingest_dlt_source import _row_content_hash

TABLE = "asana_test"
TOKEN = "SECRET-asana-token-123"
ROW_COLUMNS = {"id", "title", "content", "url", "_deleted", NODE_SET_COLUMN}
REASONS = {"failed_auth", "failed_rate_limit", "failed_budget", "failed_unavailable"}
# What one task costs to read: the task, its comments and its subtasks.
TASK_REQUESTS = 3
# What a run spends before it renders a task of a one project selection: the project, its
# events and the task listing.
LISTING_REQUESTS = 3


def _workspace(tasks: int = 3) -> FakeAsana:
    fake = FakeAsana()
    fake.add_project("P1")
    for gid in range(1, tasks + 1):
        fake.add_task(str(gid), f"Task {gid}", notes=f"Body {gid}.")
    return fake


def _sync(service, state: dict, **kwargs) -> tuple[list[dict], dict]:
    stats = new_stats()
    kwargs.setdefault("project_gids", ["P1"])
    rows = list(_iter_rows(service, state, stats, **kwargs))
    return rows, stats


def _tasks(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows if r["id"].startswith("task:") and not r["_deleted"]}


def _tombstones(rows: list[dict]) -> set[str]:
    return {r["id"] for r in rows if r["_deleted"]}


def _fetched(fake: FakeAsana) -> list[str]:
    """The gids of the tasks read to render them, in the order their task answers were asked for."""
    return [
        path.split("/")[2]
        for path, params in fake.calls
        if re.fullmatch(r"/tasks/\w+", path) and "name" in params["opt_fields"]
    ]


def _checked(fake: FakeAsana) -> list[str]:
    """The gids of the missing tasks asked for their memberships, in order."""
    return [
        path.split("/")[2]
        for path, params in fake.calls
        if re.fullmatch(r"/tasks/\w+", path) and params["opt_fields"] == "memberships.project.gid"
    ]


def _reasons(stats: dict) -> set[str]:
    return {key for key in REASONS if key in stats}


# --- rendering ---


def test_a_task_renders_its_fields_subtasks_and_comments():
    task = {
        "name": "Write launch post",
        "notes": "Draft, then review with marketing.\n",
        "completed": False,
        "due_on": "2026-10-20",
        "assignee": {"name": "Ada"},
        "memberships": [
            {"project": {"gid": "P1", "name": "Website launch"}, "section": {"name": "Doing"}},
            {"project": {"gid": "P2", "name": "Blog"}, "section": None},
        ],
        "custom_fields": [
            {"name": "Priority", "display_value": "High"},
            {"name": "Estimate", "display_value": None},  # an unset field is left out
        ],
    }
    comments = [{"text": "Looks good", "created_by": {"name": "Grace"}}, {"text": "Shipping"}]
    subtasks = [
        {"gid": "10", "name": "Outline", "completed": True, "notes": ""},
        {
            "gid": "11",
            "name": "Screenshots",
            "assignee": {"name": "Grace"},
            "due_on": "2026-10-18",
            "notes": "Desktop and mobile.\nLight theme only.\n",
        },
    ]

    row = render_task("1", task, comments, subtasks, ["P1", "P2"])

    assert row == {
        "id": "task:1",
        "title": "Write launch post",
        "url": "",
        "_deleted": False,
        NODE_SET_COLUMN: ["asana:P1", "asana:P2"],
        "content": (
            "Completed: no\n"
            "Assignee: Ada\n"
            "Due: 2026-10-20\n"
            "Project: Blog\n"
            "Project: Website launch (section: Doing)\n"
            "Priority: High\n"
            "\n"
            "Draft, then review with marketing.\n"
            "\n"
            "Subtasks:\n"
            "- [x] Outline\n"
            "- [ ] Screenshots\n"
            "  Assignee: Grace\n"
            "  Due: 2026-10-18\n"
            "  Notes: Desktop and mobile.\n"
            "  Light theme only.\n"
            "\n"
            "Comments:\n"
            "- Grace: Looks good\n"
            "- Unknown: Shipping"
        ),
    }


def test_shuffled_collections_render_the_same_text():
    """The text decides re-extraction, so the order Asana answers in must not matter."""
    task = {
        "memberships": [
            {"project": {"gid": gid, "name": f"Project {gid}"}, "section": {"name": "To do"}}
            for gid in ("C", "A", "B")
        ],
        "custom_fields": [
            {"name": name, "display_value": value} for name, value in zip("ZXY", "123", strict=True)
        ],
    }
    subtasks = [{"gid": str(gid), "name": f"Step {gid}"} for gid in (9, 10, 11)]
    comments = [
        {
            "gid": str(100 + i),
            "text": f"comment {i}",
            "created_at": f"2026-10-0{i + 1}T00:00:00.000Z",
        }
        for i in range(4)
    ]
    expected = render_task("1", task, comments, subtasks, "ABC")

    rng = random.Random(7)
    for _ in range(8):
        shuffled = {
            **task,
            "memberships": rng.sample(task["memberships"], 3),
            "custom_fields": rng.sample(task["custom_fields"], 3),
        }
        got = render_task("1", shuffled, rng.sample(comments, 4), rng.sample(subtasks, 3), "CBA")
        assert got == expected
    # Oldest comment first, subtasks in creation order: gid 9 before 10 although "10" < "9"
    assert expected["content"].index("comment 0") < expected["content"].index("comment 3")
    assert expected["content"].index("Step 9") < expected["content"].index("Step 10")


def test_nothing_volatile_reaches_the_text():
    base = {
        "name": "T",
        "notes": "body",
        "memberships": [],
        "modified_at": "2026-10-01T00:00:00.000Z",
        "num_likes": 0,
    }
    later = {**base, "modified_at": "2026-10-09T00:00:00.000Z", "num_likes": 7}

    assert render_task("1", base, [], [], []) == render_task("1", later, [], [], [])


def test_only_selected_projects_are_named_in_a_task():
    task = {
        "memberships": [
            {"project": {"gid": "P1", "name": "Selected"}, "section": None},
            {"project": {"gid": "P9", "name": "Not selected"}, "section": None},
        ]
    }

    content = render_task("1", task, [], [], ["P1"])["content"]

    assert "Project: Selected" in content and "Not selected" not in content


def test_a_task_with_nothing_set_is_still_a_document():
    row = render_task("1", {"memberships": [], "completed": True}, [], [], [])

    assert (row["title"], row["content"]) == ("Untitled task", "Completed: yes")


def _member(gid: str) -> dict:
    return {"project": {"gid": gid, "name": f"Project {gid}"}, "section": None}


def test_a_task_in_one_selected_project_has_that_projects_node_set():
    row = render_task("1", {"memberships": [_member("P1")]}, [], [], ["P1", "P2"])

    assert row[NODE_SET_COLUMN] == ["asana:P1"]


def test_a_task_in_two_selected_projects_has_both_sorted_whatever_the_order_asana_sends():
    """The list is hashed into the document identity: a different order would flap it."""
    forward = render_task(
        "1", {"memberships": [_member("P1"), _member("P2")]}, [], [], ["P1", "P2"]
    )
    backward = render_task(
        "1", {"memberships": [_member("P2"), _member("P1")]}, [], [], ["P2", "P1"]
    )

    assert forward[NODE_SET_COLUMN] == backward[NODE_SET_COLUMN] == ["asana:P1", "asana:P2"]
    assert _row_content_hash(forward) == _row_content_hash(backward)


def test_a_project_the_task_is_in_but_nobody_selected_is_not_in_its_node_set():
    task = {"memberships": [_member("P1"), _member("P9"), _member("P1")]}

    assert render_task("1", task, [], [], ["P1"])[NODE_SET_COLUMN] == ["asana:P1"]


def test_the_node_set_holds_gids_never_names():
    task = {"memberships": [{"project": {"gid": "P1", "name": "Secret launch"}, "section": None}]}
    renamed = {"memberships": [{"project": {"gid": "P1", "name": "Other name"}, "section": None}]}

    row, after_rename = (
        render_task("1", task, [], [], ["P1"]),
        render_task("1", renamed, [], [], ["P1"]),
    )

    assert "Secret" not in json.dumps(row[NODE_SET_COLUMN])
    assert row[NODE_SET_COLUMN] == after_rename[NODE_SET_COLUMN]


def test_a_project_row_names_only_itself():
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")

    rows, _ = _sync(fake, {}, project_gids=["P1", "P2"])

    assert {row["id"]: row[NODE_SET_COLUMN] for row in rows} == {
        "project:P1": ["asana:P1"],
        "project:P2": ["asana:P2"],
    }


def test_a_task_in_no_selected_project_has_an_empty_node_set_that_core_does_not_hash():
    """A task that left the selection between the listing and its read: the empty list is the
    shape core leaves out of the identity, so such a row hashes as one without the column."""
    row = render_task("1", {"memberships": [_member("P9")]}, [], [], ["P1"])

    assert row[NODE_SET_COLUMN] == []
    assert _row_content_hash(row) == _row_content_hash(
        {key: value for key, value in row.items() if key != NODE_SET_COLUMN}
    )


# --- the completion day ---


@pytest.mark.parametrize(
    ("completed_at", "line"),
    [
        ("2026-10-09T19:14:27.123Z", "Completed on: 2026-10-09"),
        ("2026-10-09T00:00:00.000Z", "Completed on: 2026-10-09"),
        ("2026-10-09T23:59:59.999Z", "Completed on: 2026-10-09"),
        # the UTC day, not the day of the offset the stamp happens to carry
        ("2026-10-10T01:30:00.000+02:00", "Completed on: 2026-10-09"),
    ],
)
def test_a_completed_task_names_the_utc_day_it_was_completed_and_never_the_time(completed_at, line):
    row = render_task(
        "1", {"memberships": [], "completed": True, "completed_at": completed_at}, [], [], []
    )

    assert row["content"] == f"Completed: yes\n{line}"


def test_an_open_task_renders_no_completion_day_even_if_asana_still_sends_one():
    """Reopening a task changes its text, which is right: the line goes with ``Completed: yes``."""
    done = {"memberships": [], "completed": True, "completed_at": "2026-10-09T19:14:27.123Z"}
    reopened = {"memberships": [], "completed": False, "completed_at": None}
    stale = {"memberships": [], "completed": False, "completed_at": "2026-10-09T19:14:27.123Z"}

    assert "Completed on" in render_task("1", done, [], [], [])["content"]
    assert render_task("1", reopened, [], [], [])["content"] == "Completed: no"
    assert render_task("1", stale, [], [], [])["content"] == "Completed: no"


@pytest.mark.parametrize(
    "completed_at",
    [None, "", "yesterday", "2026-13-45T00:00:00.000Z", 20261009, ["2026-10-09"], "2026-10-09"],
)
def test_a_completed_at_that_cannot_be_read_renders_no_line_and_never_raises(completed_at):
    row = render_task(
        "1", {"memberships": [], "completed": True, "completed_at": completed_at}, [], [], []
    )

    assert row["content"] == "Completed: yes"


def test_a_completed_task_without_completed_at_renders_no_line():
    assert (
        render_task("1", {"memberships": [], "completed": True}, [], [], [])["content"]
        == "Completed: yes"
    )


# --- events to documents ---


def test_event_task_gids_maps_each_event_shape():
    ref = {"gid": "t-added", "resource_type": "task"}
    events = [
        {"resource": {"gid": "s1", "resource_type": "story"}, "parent": ref},
        # an edited comment: parent is null, only resource.target names the task
        {
            "resource": {
                "gid": "s2",
                "resource_type": "story",
                "target": {"gid": "t-edited", "resource_type": "task"},
            },
            "parent": None,
        },
        {"resource": {"gid": "t-renamed", "resource_type": "task"}, "parent": None},
        # a subtask added under a task: the parent is the document to refresh
        {
            "resource": {"gid": "sub1", "resource_type": "task"},
            "parent": {"gid": "t-parent", "resource_type": "task"},
        },
        {"action": "deleted", "resource": {"gid": "t-deleted", "resource_type": "task"}},
        # ignored: a project-level story, other resource types
        {
            "resource": {"gid": "s3", "resource_type": "story"},
            "parent": {"gid": "P1", "resource_type": "project"},
        },
        {"resource": {"gid": "sec", "resource_type": "section"}},
    ]

    assert _event_task_gids(events) == {
        "t-added",
        "t-edited",
        "t-renamed",
        "t-parent",
        "t-deleted",
    }


# Asana's shapes drift: an event field that is a string or a null where an object is expected,
# and an event with no gid at all, must be skipped and never crash the run.
@pytest.mark.parametrize(
    "event",
    [
        {"resource": "x"},
        {"resource": {"resource_type": "story", "target": "t1"}, "parent": None},
        {"resource": {"resource_type": "story", "target": {"resource_type": "task"}}},
        {"resource": {"resource_type": "task"}, "parent": "t1"},
        {"resource": {"resource_type": "task", "gid": 7}},
        {"resource": None, "parent": None},
        {"resource": {"resource_type": "story"}, "parent": {"resource_type": "task"}},
        "a string",
        7,
        None,
        {},
    ],
)
def test_events_that_name_no_task_are_skipped(event):
    assert _event_task_gids([event]) == set()


def test_a_string_where_an_object_is_expected_does_not_hide_the_other_events():
    events = [
        {"resource": "x"},
        {"resource": {"gid": "t1", "resource_type": "task"}, "parent": "y"},
        {"resource": {"resource_type": "story", "target": "z"}, "parent": {"gid": "t2"}},
    ]

    assert _event_task_gids(events) == {"t1"}


def test_only_a_project_or_section_name_change_is_a_rename():
    def changed(kind, field, action="changed"):
        return {
            "action": action,
            "resource": {"gid": "x", "resource_type": kind},
            "change": {"field": field},
        }

    assert _renamed([changed("section", "name")]) is True
    assert _renamed([changed("project", "name")]) is True
    assert _renamed([changed("project", "notes")]) is False
    assert _renamed([changed("task", "name")]) is False
    assert _renamed([changed("section", "name", action="added")]) is False
    assert _renamed(["x", {"resource": "x", "change": "name"}]) is False


def test_documents_for_maps_subtasks_to_their_parent_and_drops_the_rest():
    swept, subtask_parents = {"1", "2"}, {"10": "1", "30": "3"}  # task 3 left the selection

    assert _documents_for({"1", "10", "30", "unknown"}, swept, subtask_parents) == {"1"}
    assert _documents_for(set(), swept, subtask_parents) == set()


# --- first sync and incremental sync ---


def test_a_first_sync_emits_the_project_and_every_task_with_its_comments_and_subtasks():
    fake = _workspace(tasks=2)
    fake.tasks["1"].update(assignee="Ada", comments=[])
    fake.add_comment("1", "Grace", "Use the new palette")
    fake.add_task("10", "Pick fonts", parent="1")
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert [row["id"] for row in rows] == ["project:P1", "task:1", "task:2"]
    assert all(set(row) == ROW_COLUMNS for row in rows), "exactly the document columns"
    task = _tasks(rows)["task:1"]
    assert task["url"] == "https://app.asana.com/t/1"
    assert "Assignee: Ada" in task["content"]
    assert "Project: Website launch (section: To do)" in task["content"]
    assert "- [ ] Pick fonts" in task["content"]
    assert "- Grace: Use the new palette" in task["content"]
    assert "added it" not in task["content"], "system stories are not comments"
    assert "2026-10" not in task["content"], "no timestamps in the text"
    assert (rows[0]["title"], rows[0]["content"]) == ("Website launch", "Ship the site.")
    assert state["v"] == STATE_VERSION == 1
    assert state["selection"] == ["P1"]
    assert state["known_ids"] == ["project:P1", "task:1", "task:2"]
    assert state["subtasks"] == {"10": "1"}, "remembered so a subtask event finds task 1"
    assert state["projects"]["P1"]["cursor"] == fake.tasks["2"]["modified_at"]
    assert state["projects"]["P1"]["sync"] == fake.tokens["P1"]
    assert (state["retry"], state["pending"]) == ([], [])
    assert stats == {**new_stats(), "scanned": 2, "requests": LISTING_REQUESTS + 2 * TASK_REQUESTS}


# --- timestamp forms ---
# Timestamps are compared as strings, which orders them only in one form: UTC, three decimals,
# ``Z``. Asana sends that form; these are the other valid ISO spellings of the same instants.
_FORMS = {
    "no milliseconds": lambda sec: f"2026-10-01T09:00:{sec:02d}Z",
    "microseconds": lambda sec: f"2026-10-01T09:00:{sec:02d}.000000Z",
    "utc offset": lambda sec: f"2026-10-01T09:00:{sec:02d}.000+00:00",
    "east offset": lambda sec: f"2026-10-01T11:00:{sec:02d}.000+02:00",
    "west offset": lambda sec: f"2026-10-01T04:00:{sec:02d}.000-05:00",
}


def _canonical(sec: int) -> str:
    return f"2026-10-01T09:00:{sec:02d}.000Z"


def _stamped_runs(form) -> tuple[dict, dict[str, str], dict, tuple[list[str], int]]:
    """Three runs over tasks whose ``modified_at`` instants are written in ``form``: a first
    sync, one after tasks 2 and 4 changed (listing only: no event) and one with no change.

    Returns the state after the first run, the tasks the second run rendered, its state, and
    what the third rendered with its request count.
    """
    fake = FakeAsana()
    fake.add_project("P1")
    for gid in range(1, 5):
        fake.add_task(str(gid), notes=f"Body {gid}.", modified_at=form(gid))
    state: dict = {}
    _sync(fake, state)
    first = copy.deepcopy(state)
    for gid, sec in (("2", 30), ("4", 31)):
        fake.tasks[gid].update(notes=f"Edit of {gid}.", modified_at=form(sec))
    rows, _ = _sync(fake, state)
    second = copy.deepcopy(state)
    changed = {gid: row["content"] for gid, row in _tasks(rows).items()}
    rows, stats = _sync(fake, state)
    return first, changed, second, (sorted(_tasks(rows)), stats["requests"])


@pytest.mark.parametrize("form", _FORMS.values(), ids=_FORMS)
def test_the_same_instants_in_another_form_give_the_same_state_and_the_same_changes(form):
    expected = _stamped_runs(_canonical)

    assert _stamped_runs(form) == expected

    first, changed, second, (unchanged, requests) = expected
    assert first["projects"]["P1"]["cursor"] == _canonical(4)
    assert second["projects"]["P1"]["cursor"] == _canonical(31)
    assert second["projects"]["P1"]["seen"] == {
        gid: _canonical(sec) for gid, sec in (("1", 1), ("2", 30), ("3", 3), ("4", 31))
    }
    assert sorted(changed) == ["task:2", "task:4"]
    assert "Edit of 2." in changed["task:2"] and "Edit of 4." in changed["task:4"]
    assert (unchanged, requests) == ([], LISTING_REQUESTS), "an unchanged run renders nothing"


def test_a_task_changed_inside_the_overlap_window_is_seen_whatever_the_offset_of_its_stamp():
    """The instant is 09:00:20 UTC, after the window start (08:59:02). Written as 04:00:20-05:00
    its string sorts before the start, so a string comparison would miss the edit."""
    fake = FakeAsana()
    fake.add_project("P1")
    fake.add_task("1", modified_at=_canonical(1))
    fake.add_task("2", modified_at=_canonical(2))
    state: dict = {}
    _sync(fake, state)
    fake.tasks["1"].update(notes="Edited.", modified_at=_FORMS["west offset"](20))

    rows, _ = _sync(fake, state)

    assert "Edited." in _tasks(rows)["task:1"]["content"]
    assert state["projects"]["P1"]["cursor"] == _canonical(20)


def test_a_stamp_is_kept_in_asanas_form_and_read_back_unchanged():
    assert _overlap_start("") == ""
    assert _stamp(_instant("2026-10-09T19:14:27.123Z")) == "2026-10-09T19:14:27.123Z"
    assert _stamp(_instant("2026-10-09T21:14:27.123456+02:00")) == "2026-10-09T19:14:27.123Z"
    assert _overlap_start("2026-10-09T19:14:27.123Z") == "2026-10-09T19:13:27.123Z"
    with pytest.raises(ValueError, match="without an offset"):
        _stamp(_instant("2026-10-09T19:14:27"))


def test_an_unchanged_second_sync_reads_no_task():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    before = copy.deepcopy(state)
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert [row["id"] for row in rows] == ["project:P1"], "only the project, one request"
    assert _fetched(fake) == []
    assert (stats["scanned"], stats["skipped"], stats["failed"]) == (3, 3, 0)
    assert state["known_ids"] == before["known_ids"]
    assert stats["requests"] == LISTING_REQUESTS


def test_an_edited_task_is_rendered_again_and_only_that_one():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2", notes="Rewritten.")
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:2"} and "Rewritten." in _tasks(rows)["task:2"]["content"]
    assert _fetched(fake) == ["2"]
    assert (stats["scanned"], stats["skipped"]) == (3, 2)


# Asana does not move modified_at for a comment edit or a new comment: the Events API is the only
# way to see them, so a connector that trusted modified_at alone would keep the old text.
@pytest.mark.parametrize("change", ["edit", "add"])
def test_a_comment_is_seen_through_events_while_modified_at_stays_put(change):
    fake = _workspace()
    fake.add_comment("1", "Grace", "first")
    state: dict = {}
    _sync(fake, state)
    stamp = fake.tasks["1"]["modified_at"]
    if change == "edit":
        fake.edit_comment("1", 0, "edited")
    else:
        fake.add_comment("1", "Ada", "second")

    rows, _ = _sync(fake, state)

    assert fake.tasks["1"]["modified_at"] == stamp
    assert set(_tasks(rows)) == {"task:1"}
    assert ("edited" if change == "edit" else "- Ada: second") in _tasks(rows)["task:1"]["content"]


def test_a_renamed_subtask_refreshes_its_parent_through_the_subtask_index():
    fake = _workspace(tasks=2)
    fake.add_task("10", "Pick fonts", parent="1")
    state: dict = {}
    _sync(fake, state)
    fake.rename_subtask("10", "Pick two fonts")

    rows, _ = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1"}
    assert "- [ ] Pick two fonts" in _tasks(rows)["task:1"]["content"]


def test_the_subtask_index_is_dropped_with_its_parent():
    fake = _workspace(tasks=2)
    fake.add_task("10", "Pick fonts", parent="1")
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("10")
    fake.delete_task("1")

    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == {"task:1"}
    assert state["subtasks"] == {}


@pytest.mark.parametrize("kind", ["project", "section"])
def test_a_renamed_project_or_section_renders_every_task_of_the_project_again(kind):
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    stamps = {gid: task["modified_at"] for gid, task in fake.tasks.items()}
    fake.rename(kind, "P1", "Renamed", section="To do")

    rows, _ = _sync(fake, state)

    assert {gid: task["modified_at"] for gid, task in fake.tasks.items()} == stamps
    assert set(_tasks(rows)) == {"task:1", "task:2", "task:3"}
    assert all("Renamed" in row["content"] for row in _tasks(rows).values())


def test_an_edited_project_description_renders_only_the_project():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.projects["P1"]["notes"] = "A new brief."

    rows, _ = _sync(fake, state)

    assert [row["id"] for row in rows] == ["project:P1"]
    assert rows[0]["content"] == "A new brief."


def test_events_are_followed_across_every_page_with_no_cap():
    fake = _workspace(tasks=5)
    state: dict = {}
    _sync(fake, state)
    fake.events_page_size = 1
    for gid in "12345":
        fake.add_comment(gid, "Ada", f"note {gid}")
    fake.calls.clear()

    rows, _ = _sync(fake, state)

    assert len(fake.requested("/events")) == 5
    assert set(_tasks(rows)) == {f"task:{gid}" for gid in "12345"}


def test_an_expired_sync_token_renders_the_project_again_and_stores_the_fresh_one():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.expire_token("P1")

    rows, _ = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1", "task:2", "task:3"}
    assert state["projects"]["P1"]["sync"] == fake.tokens["P1"]
    assert _tasks(_sync(fake, state)[0]) == {}, "and the next run is quiet again"


def test_an_events_answer_without_a_sync_token_is_an_error():
    fake = _workspace()
    fake.answers["/events"] = {"data": [], "has_more": False}

    with pytest.raises(AsanaResponseError, match="no sync token"):
        _sync(fake, {})


def _state_with_sync(fake: FakeAsana, token: str) -> dict:
    return {"projects": {"P1": {"cursor": fake.stamp(), "sync": token, "seen": {}}}}


def test_an_events_feed_whose_token_does_not_advance_raises_instead_of_looping():
    fake = _workspace()
    fake.answers["/events"] = {"data": [], "sync": "same", "has_more": True}
    state = _state_with_sync(fake, "same")

    with pytest.raises(AsanaResponseError, match="sync token came back twice"):
        _sync(fake, state)


def test_an_events_feed_whose_first_token_is_the_stored_one_raises():
    fake = _workspace()
    fake.event_feed = [("S", True), ("A", False)]
    state = _state_with_sync(fake, "S")

    with pytest.raises(AsanaResponseError, match="sync token came back twice"):
        _sync(fake, state)

    assert len(fake.requested("/events")) == 1


@pytest.mark.parametrize(
    ("cycle", "requests"), [(["A", "B"], 3), (["A", "B", "C"], 4)], ids=["A-B-A", "A-B-C-A"]
)
def test_an_events_feed_that_cycles_through_its_tokens_raises_instead_of_looping(cycle, requests):
    """Each page's token differs from the one before it, so only remembering them all finds it."""
    fake = _workspace()
    fake.event_feed = [(token, True) for token in cycle * 100]
    state = _state_with_sync(fake, "S")
    before = copy.deepcopy(state)

    with pytest.raises(AsanaResponseError, match="Asana /events: a sync token came back twice"):
        _sync(fake, state)

    assert len(fake.requested("/events")) == requests
    assert state == before


def test_an_events_feed_of_many_distinct_tokens_is_read_in_full():
    fake = _workspace()
    fake.event_feed = [(f"t{n}", True) for n in range(49)] + [("end", False)]
    state = _state_with_sync(fake, "S")

    _sync(fake, state)

    assert len(fake.requested("/events")) == 50
    assert state["projects"]["P1"]["sync"] == "end"


def test_a_last_page_that_repeats_the_token_is_not_a_cycle():
    """A quiet feed answers its own token with nothing more to read."""
    fake = _workspace()
    fake.event_feed = [("S", False)]
    state = _state_with_sync(fake, "S")

    _sync(fake, state)

    assert state["projects"]["P1"]["sync"] == "S"


def test_a_task_in_two_selected_projects_is_one_document_naming_both():
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1", "P2"))
    fake.add_task("2", projects=("P2",))

    rows, _ = _sync(fake, {}, project_gids=["P1", "P2"])

    assert [row["id"] for row in rows] == ["project:P1", "project:P2", "task:1", "task:2"]
    assert _tasks(rows)["task:1"]["content"].splitlines()[1:3] == [
        "Project: Blog (section: To do)",
        "Project: Website launch (section: To do)",
    ]


def test_a_task_new_to_the_selection_with_an_old_timestamp_is_ingested():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.add_task("9", modified_at="2000-01-01T00:00:00.000Z")

    rows, _ = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:9"}


# --- changing the selection ---


def test_deselecting_a_project_rewrites_the_tasks_that_named_it_and_forgets_the_rest():
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", "Shared", projects=("P1", "P2"))
    fake.add_task("2", "Only in P1", projects=("P1",))
    fake.add_task("3", "Only in P2", projects=("P2",))
    state: dict = {}
    _sync(fake, state, project_gids=["P1", "P2"])
    fake.calls.clear()

    rows, stats = _sync(fake, state, project_gids=["P1"])

    # task 1 is still listed under P1, but its document still names P2: it is rendered again
    assert _fetched(fake) == ["1"]
    assert "Blog" not in _tasks(rows)["task:1"]["content"]
    assert _tombstones(rows) == {"task:3", "project:P2"}
    assert stats["deleted"] == 2
    assert state["selection"] == ["P1"] and state["multi"] == {}
    assert state["known_ids"] == ["project:P1", "task:1", "task:2"]


def test_deselecting_a_project_changes_the_node_set_of_the_task_that_stays_in_another():
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", "Shared", projects=("P1", "P2"))
    fake.add_task("2", "Only in P1", projects=("P1",))
    state: dict = {}
    first, _ = _sync(fake, state, project_gids=["P1", "P2"])
    assert _tasks(first)["task:1"][NODE_SET_COLUMN] == ["asana:P1", "asana:P2"]
    fake.calls.clear()

    rows, _ = _sync(fake, state, project_gids=["P1"])

    # the task's modified_at never moved, yet it is read and emitted again with the new set,
    # and the task that never named P2 is not touched
    assert _fetched(fake) == ["1"]
    assert _tasks(rows)["task:1"][NODE_SET_COLUMN] == ["asana:P1"]
    assert _row_content_hash(_tasks(rows)["task:1"]) != _row_content_hash(_tasks(first)["task:1"])
    assert "task:2" not in _tasks(rows)


def test_selecting_a_project_adds_its_gid_to_the_node_set_of_tasks_already_synced():
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1", "P2"))
    state: dict = {}
    first, _ = _sync(fake, state, project_gids=["P1"])
    assert _tasks(first)["task:1"][NODE_SET_COLUMN] == ["asana:P1"]

    rows, _ = _sync(fake, state, project_gids=["P1", "P2"])

    assert _tasks(rows)["task:1"][NODE_SET_COLUMN] == ["asana:P1", "asana:P2"]


def test_a_renamed_project_changes_the_text_of_its_tasks_but_not_their_node_sets():
    fake = _workspace(tasks=1)
    state: dict = {}
    first, _ = _sync(fake, state)

    fake.rename("project", "P1", "Relaunch")
    rows, _ = _sync(fake, state)

    assert _tasks(rows)["task:1"]["content"] != _tasks(first)["task:1"]["content"]
    assert _tasks(rows)["task:1"][NODE_SET_COLUMN] == _tasks(first)["task:1"][NODE_SET_COLUMN]
    assert next(r for r in rows if r["id"] == "project:P1")[NODE_SET_COLUMN] == ["asana:P1"]


def test_completing_and_reopening_a_task_changes_its_text_each_time():
    fake = _workspace(tasks=1)
    state: dict = {}
    _sync(fake, state)

    fake.edit_task("1", completed=True, completed_at="2026-10-09T19:14:27.123Z")
    done, _ = _sync(fake, state)
    fake.edit_task("1", completed=False, completed_at=None)
    reopened, _ = _sync(fake, state)

    assert _tasks(done)["task:1"]["content"].startswith("Completed: yes\nCompleted on: 2026-10-09")
    assert _tasks(reopened)["task:1"]["content"].startswith("Completed: no\n")
    assert "Completed on" not in _tasks(reopened)["task:1"]["content"]


def test_selecting_a_project_adds_its_name_to_tasks_already_synced_from_another():
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1", "P2"))
    fake.add_task("2", projects=("P2",))
    state: dict = {}
    first, _ = _sync(fake, state, project_gids=["P1"])
    assert "Blog" not in _tasks(first)["task:1"]["content"]

    rows, _ = _sync(fake, state, project_gids=["P1", "P2"])

    assert "Project: Blog" in _tasks(rows)["task:1"]["content"]
    assert "task:2" in _tasks(rows)
    assert state["multi"] == {"1": ["P1", "P2"]}


# --- a listing cut by a rate limit resumes ---


def _three_projects(tasks_each: int = 2) -> FakeAsana:
    """P1, P2 and P3 with ``tasks_each`` tasks of their own (gids ``<project number><n>``)."""
    fake = FakeAsana()
    for number in (1, 2, 3):
        fake.add_project(f"P{number}", f"Project {number}", notes=f"About {number}.")
        for n in range(1, tasks_each + 1):
            fake.add_task(f"{number}{n}", f"Task {number}{n}", projects=(f"P{number}",))
    return fake


def _projects_requested(fake: FakeAsana) -> list[str]:
    return [path.split("/")[2] for path, _ in fake.calls if re.fullmatch(r"/projects/\w+", path)]


SELECTION = {"project_gids": ["P1", "P2", "P3"]}


def test_a_rate_limit_inside_the_listing_keeps_the_projects_already_listed():
    fake = _three_projects()
    fake.fail("/tasks", 429, after=2)  # the third project's listing page
    state: dict = {}

    rows, stats = _sync(fake, state, **SELECTION)

    # the two finished projects go out as rows; no task is read and nothing is forgotten
    assert [row["id"] for row in rows] == ["project:P1", "project:P2"]
    assert (stats["failed"], stats["failed_rate_limit"], stats["deleted"]) == (1, 1, 0)
    assert state["listing"]["selection"] == ["P1", "P2", "P3"]
    assert sorted(state["listing"]["done"]) == ["P1", "P2"]
    assert state["listing"]["done"]["P1"]["next"]["tasks"] == ["11", "12"]
    assert state["known_ids"] == ["project:P1", "project:P2"]
    # nothing but the listing moved: the pass is not complete
    assert not {"projects", "selection", "pending", "retry"} & state.keys()
    fake.heal()
    fake.calls.clear()

    rows, stats = _sync(fake, state, **SELECTION)

    assert _projects_requested(fake) == ["P3"], "the finished projects are not listed again"
    assert rows[0]["id"] == "project:P3"
    assert set(_tasks(rows)) == {f"task:{n}{m}" for n in (1, 2, 3) for m in (1, 2)}
    assert (stats["failed"], stats["scanned"], stats["deleted"]) == (0, 6, 0)
    assert "listing" not in state, "the pass is complete"
    assert (
        sorted(state["projects"]) == ["P1", "P2", "P3"]
        and state["selection"] == SELECTION["project_gids"]
    )


def test_a_pass_cut_inside_the_listing_forgets_nothing_and_a_complete_one_does():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.delete_task("11")
    fake.delete_task("31")
    fake.fail("/tasks", 429, after=1)  # P1 lists, P2 is cut

    rows, stats = _sync(fake, state, **SELECTION)

    assert _tombstones(rows) == set() and stats["deleted"] == 0
    assert "task:11" in state["known_ids"] and "task:31" in state["known_ids"]
    fake.heal()

    rows, stats = _sync(fake, state, **SELECTION)

    assert _tombstones(rows) == {"task:11", "task:31"} and stats["deleted"] == 2
    assert "listing" not in state


def test_a_pass_cut_inside_the_listing_renders_no_task_even_one_a_finished_project_changed():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.edit_task("11", notes="Rewritten.")
    fake.fail("/tasks", 429, after=1)
    fake.calls.clear()

    rows, _ = _sync(fake, state, **SELECTION)

    assert _tasks(rows) == {} and _fetched(fake) == []
    fake.heal()

    rows, _ = _sync(fake, state, **SELECTION)

    # P2 was cut after its events were read, so its token is stale and it is read in full: more
    # than needed, never less. P3 was not touched.
    assert {"task:11"} <= set(_tasks(rows)) and not {"task:12", "task:31", "task:32"} & set(
        _tasks(rows)
    ), "the change the finished project saw is not lost"


def test_a_run_cut_before_it_finished_any_project_moves_nothing():
    fake = _three_projects()
    fake.fail("/projects", 429)
    state: dict = {}

    rows, stats = _sync(fake, state, **SELECTION)

    assert rows == [] and state == {} and stats["failed_rate_limit"] == 1


def test_a_changed_selection_starts_the_listing_pass_over():
    fake = _three_projects()
    fake.fail("/tasks", 429, after=2)
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.heal()
    fake.calls.clear()

    rows, _ = _sync(fake, state, project_gids=["P1", "P3"])

    assert _projects_requested(fake) == ["P1", "P3"], "P1 is listed again for the new selection"
    assert set(_tasks(rows)) == {f"task:{n}{m}" for n in (1, 3) for m in (1, 2)}
    # P2's row went out in the cut pass: the completed pass for the new selection forgets it
    assert _tombstones(rows) == {"project:P2"}
    assert "listing" not in state and state["selection"] == ["P1", "P3"]


def test_a_cut_pass_of_another_selection_replaces_the_kept_one():
    fake = _three_projects()
    fake.fail("/tasks", 429, after=1)
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.heal()
    fake.fail("/tasks", 429, after=1)

    _sync(fake, state, project_gids=["P1", "P2"])

    assert state["listing"]["selection"] == ["P1", "P2"]
    assert sorted(state["listing"]["done"]) == ["P1"]


def test_the_listing_state_holds_ids_and_timestamps_never_text():
    fake = _three_projects()
    fake.fail("/tasks", 429, after=2)
    state: dict = {}
    _sync(fake, state, **SELECTION)

    stored = json.dumps(state["listing"])

    assert "Project" not in stored and "About" not in stored and "Task" not in stored


def test_a_run_that_raises_after_listing_a_project_leaves_the_state_as_it_was():
    fake = _three_projects()
    fake.fail("/tasks", 429, after=1)
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.heal()
    fake.answers["/projects/P3"] = {"data": "not an object"}
    before = copy.deepcopy(state)

    with pytest.raises(AsanaResponseError):
        _sync(fake, state, **SELECTION)

    assert state == before


def test_a_complete_listing_in_one_run_leaves_no_listing_state_and_costs_what_it_did():
    fake = _three_projects()
    state: dict = {}

    rows, stats = _sync(fake, state, **SELECTION)

    assert "listing" not in state
    assert stats["requests"] == 3 * LISTING_REQUESTS + 3 * 2 * TASK_REQUESTS
    assert [row["id"] for row in rows][:3] == ["project:P1", "project:P2", "project:P3"]


# --- a task the listing lacks is forgotten only when Asana says it is gone ---


def _two_projects_with_x() -> FakeAsana:
    """A=P1 holds task a1; B=P2 holds task X and b1."""
    fake = FakeAsana()
    fake.add_project("P1", "A")
    fake.add_project("P2", "B")
    fake.add_task("a1", "In A", projects=("P1",))
    fake.add_task("X", "Mover", projects=("P2",))
    fake.add_task("b1", "In B", projects=("P2",))
    return fake


BOTH = {"project_gids": ["P1", "P2"]}


def _cut_after_a(tmp_path, fake: FakeAsana) -> None:
    """A synced pass, then a run that lists A and is cut on B's listing."""
    _run(tmp_path, _source(fake, **BOTH))
    fake.fail("/tasks", 429, after=1)
    cut = _source(fake, **BOTH)
    _run(tmp_path, cut)
    assert cut.cognee_sync_stats["failed_rate_limit"] == 1
    fake.heal()
    fake.calls.clear()


def test_a_task_that_moves_between_the_cut_run_and_the_resuming_run_is_never_forgotten(tmp_path):
    """X leaves B for A after A was listed (old list: no X) and before B is listed (fresh list:
    no X). It is in neither, yet it exists: it is asked for its memberships, and kept."""
    fake = _two_projects_with_x()
    _cut_after_a(tmp_path, fake)
    fake.move_task("X", ["P1"])

    source = _source(fake, **BOTH)
    staged = _run(tmp_path, source)

    assert "task:X" in staged and source.cognee_sync_stats["deleted"] == 0
    assert _checked(fake) == ["X"]
    # and the move reaches the text on the next pass, through A's events
    again = _run(tmp_path, _source(fake, **BOTH))
    assert "Project: A" in again["task:X"] and "Project: B" not in again["task:X"]


def test_a_task_deleted_in_the_cut_pass_is_forgotten_because_asana_says_404(tmp_path):
    fake = _two_projects_with_x()
    _cut_after_a(tmp_path, fake)
    fake.delete_task("X")

    source = _source(fake, **BOTH)
    staged = _run(tmp_path, source)

    assert _checked(fake) == ["X"], "the listing alone is not enough to forget it"
    assert "task:X" not in staged and source.cognee_sync_stats["deleted"] == 1
    assert {"task:a1", "task:b1"} <= set(staged)


def test_a_task_that_left_the_selection_is_forgotten_because_no_project_is_selected():
    fake = _two_projects_with_x()
    fake.add_project("P9", "Elsewhere")
    state: dict = {}
    _sync(fake, state, **BOTH)
    fake.move_task("X", ["P9"])
    fake.calls.clear()

    rows, _ = _sync(fake, state, **BOTH)

    assert _tombstones(rows) == {"task:X"} and _checked(fake) == ["X"]


def test_a_check_cut_by_a_rate_limit_forgets_nothing_and_a_later_pass_does():
    fake = _two_projects_with_x()
    state: dict = {}
    _sync(fake, state, **BOTH)
    fake.delete_task("X")
    fake.fail("/tasks/X", 429)

    rows, stats = _sync(fake, state, **BOTH)

    assert _tombstones(rows) == set() and stats["failed_rate_limit"] == 1
    assert "task:X" in state["known_ids"]
    fake.heal()

    rows, stats = _sync(fake, state, **BOTH)

    assert _tombstones(rows) == {"task:X"} and stats["failed"] == 0


def test_a_check_cut_by_the_budget_keeps_the_task_known_and_a_later_run_forgets_it():
    fake = _workspace(tasks=3)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")

    # the listing is the whole budget: the check of task 2 is the request it cannot pay
    rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS)

    assert _tombstones(rows) == set() and stats["failed_budget"] == 1
    assert "task:2" in state["known_ids"]

    rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS)

    assert _tombstones(rows) == {"task:2"} and stats["failed"] == 0
    assert "task:2" not in state["known_ids"]


def test_a_resumed_pass_with_no_missing_task_makes_no_extra_request(tmp_path):
    fake = _two_projects_with_x()
    _cut_after_a(tmp_path, fake)

    source = _source(fake, **BOTH)
    _run(tmp_path, source)

    assert _checked(fake) == [] and source.cognee_sync_stats["deleted"] == 0
    assert len([1 for path, _ in fake.calls if path == "/tasks"]) == 1, "only B is listed"


def test_a_missing_task_is_checked_in_a_pass_listed_in_one_run_too():
    fake = _two_projects_with_x()
    state: dict = {}
    _sync(fake, state, **BOTH)
    fake.delete_task("X")
    fake.calls.clear()

    rows, _ = _sync(fake, state, **BOTH)

    assert _tombstones(rows) == {"task:X"} and _checked(fake) == ["X"]


def test_a_task_that_moves_while_the_pass_is_listing_is_not_forgotten():
    """X is in B when A is listed and in A when B is listed: the listing never shows it."""
    fake = _two_projects_with_x()
    state: dict = {}
    _sync(fake, state, **BOTH)
    original, listings = fake.respond, []

    def respond(path, params=None):
        answer = original(path, params)
        if path == "/tasks":
            listings.append(params["project"])
            if len(listings) == 1:  # A has just been listed
                fake.move_task("X", ["P1"])
        return answer

    fake.respond = respond
    rows, _ = _sync(fake, state, **BOTH)

    assert listings == ["P1", "P2"] and _tombstones(rows) == set()
    assert _checked(fake) == ["X"] and "task:X" in state["known_ids"]
    rows, _ = _sync(fake, state, **BOTH)
    assert "task:X" in _tasks(rows), "A's events render the moved task on the next pass"


def test_the_check_of_a_live_task_is_a_request_of_the_budget_too():
    """Two tasks move while the pass lists: the listing is six requests, so a budget of seven
    pays for one check and stops at the other, which stays unasked."""
    fake = _two_projects_with_x()
    fake.add_task("Y", "Second mover", projects=("P2",))
    state: dict = {}
    _sync(fake, state, **BOTH)
    original, listings = fake.respond, []

    def respond(path, params=None):
        answer = original(path, params)
        if path == "/tasks":
            listings.append(params["project"])
            if len(listings) == 1:
                fake.move_task("X", ["P1"])
                fake.move_task("Y", ["P1"])
        return answer

    fake.respond = respond
    fake.calls.clear()

    rows, stats = _sync(fake, state, max_requests=2 * LISTING_REQUESTS + 1, **BOTH)

    assert _tombstones(rows) == set() and stats["failed_budget"] == 1
    assert stats["requests"] == 2 * LISTING_REQUESTS + 1 and len(_checked(fake)) == 1


def test_a_page_that_shifts_on_resume_does_not_make_a_live_task_look_gone():
    """The offset stands for a position: a task removed from the first page while the run was cut
    moves every later one up, and the resumed pages skip one that is still there."""
    fake = _long_project(tasks=6, page_size=2)
    state: dict = {}
    _sync(fake, state)
    fake.fail("/tasks", 429, after=1)  # page 1 is listed, page 2 is cut
    _sync(fake, state)
    fake.heal()
    fake.delete_task("1")
    fake.calls.clear()

    rows, _ = _sync(fake, state)

    assert "task:3" not in _tombstones(rows) and "task:3" in state["known_ids"]
    assert "3" in _checked(fake), "task 3 was skipped by the shifted page and asked for"
    # task 1 was already listed on the first page, so only the next pass forgets it
    rows, _ = _sync(fake, state)
    assert _tombstones(rows) == {"task:1"}


def test_a_task_dragged_to_the_top_of_the_project_list_is_not_forgotten():
    fake = _long_project(tasks=6, page_size=2)
    state: dict = {}
    _sync(fake, state)
    fake.fail("/tasks", 429, after=1)
    _sync(fake, state)
    fake.heal()
    # task 5 is moved to the front: every task before it shifts one place down
    fake.tasks = {"5": fake.tasks["5"], **{k: v for k, v in fake.tasks.items() if k != "5"}}
    fake.calls.clear()

    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == set()
    assert _checked(fake), "a task the shifted pages skipped is asked for, not assumed gone"
    assert set(state["known_ids"]) >= {f"task:{n}" for n in range(1, 7)}


def test_a_listed_project_is_not_read_again_and_a_404_on_the_one_left_does_not_raise():
    """P2 is kept as listed by a cut run, so its 404 meets nothing. P3, the project the run was
    cut in, is read again and answers 404: unresolved, and the pass completes."""
    fake = _three_projects()
    fake.fail("/tasks", 429, after=2)  # P1 and P2 list, P3 is cut
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.heal()
    fake.fail("/projects/P2", 404)
    fake.fail("/projects/P3", 404)
    fake.calls.clear()

    _, stats = _sync(fake, state, **SELECTION)

    assert stats["unresolved"] == 1 and stats["failed"] == 0 and "listing" not in state
    assert [path for path, _ in fake.calls if path == "/projects/P2"] == []
    assert {"task:21", "task:22"} <= set(state["known_ids"])


def test_checks_are_requests_of_the_budget():
    fake = _workspace(tasks=3)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("1")
    fake.delete_task("2")
    fake.calls.clear()

    rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS + 1)

    assert _tombstones(rows) == {"task:1"} and stats["failed_budget"] == 1
    assert stats["requests"] == LISTING_REQUESTS + 1 and _checked(fake) == ["1"]
    assert "task:2" in state["known_ids"], "the check it could not pay for stays known"


@pytest.mark.parametrize(("budget", "runs_needed"), [(3, 5), (8, 2), (10, 2), (11, 1)])
def test_a_deleted_task_is_forgotten_after_the_runs_the_budget_needs(budget, runs_needed):
    """Two projects of six tasks (five listing requests each at two a page, ten a pass, then one
    check): the listing and the check are both paid from the budget, so the runs it takes are
    pinned. At 3 the pass spans runs and the check gets its own; at 10 the listing is the whole
    budget and the next run starts at the check; at 11 one run does it all."""
    fake = FakeAsana(page_size=2)
    for number in (1, 2):
        fake.add_project(f"P{number}", f"Project {number}")
        for n in range(1, 7):
            fake.add_task(f"{number}{n}", projects=(f"P{number}",))
    state: dict = {}
    both = {"project_gids": ["P1", "P2"]}
    for _ in range(40):
        _sync(fake, state, max_requests=100, **both)
        if "listing" not in state:
            break
    fake.delete_task("23")
    forgotten: set[str] = set()

    for runs in range(1, 60):
        rows, stats = _sync(fake, state, max_requests=budget, **both)
        assert stats["requests"] <= budget
        forgotten |= _tombstones(rows)
        if forgotten:
            break

    assert forgotten == {"task:23"} and runs == runs_needed


def test_forgetting_more_tasks_than_one_run_can_check_takes_several_runs():
    """Deselecting a project makes all its tasks missing: each is checked, and every verdict that
    was reached stays, so the work is spread over runs instead of starting over. Thirteen tasks
    at nine requests a run (six of them the listing) take five runs."""
    fake = _three_projects(tasks_each=13)
    state: dict = {}
    _sync(fake, state, **SELECTION)
    forgotten: set[str] = set()

    for runs in range(1, 40):
        rows, stats = _sync(fake, state, max_requests=9, project_gids=["P1", "P2"])
        forgotten |= _tombstones(rows)
        if stats["failed"] == 0:
            break

    assert forgotten == {"project:P3"} | {f"task:3{n}" for n in range(1, 14)}
    assert runs == 5 and stats["failed"] == 0


# --- many tasks moved at once: every answer counts as progress ---


def _bulk_move(movers: int, budget: int) -> tuple[FakeAsana, dict]:
    """PB holds ``movers`` tasks and ``zdead``, PA holds ``a``. A synced pass, then a run cut
    after PA was listed; then the movers go PB to PA, ``zdead`` is deleted and ``LATER`` is new."""
    fake = FakeAsana()
    fake.add_project("PA")
    fake.add_project("PB")
    fake.add_task("a", projects=("PA",))
    gids = [f"m{n:04d}" for n in range(movers)]
    for gid in gids:
        fake.add_task(gid, projects=("PB",))
    fake.add_task("zdead", projects=("PB",))
    state: dict = {}
    _sync(fake, state, max_requests=100_000, project_gids=["PA", "PB"])
    fake.fail("/projects/PB", 429, times=1)
    _sync(fake, state, max_requests=budget, project_gids=["PA", "PB"])
    assert list(state["listing"]["done"]) == ["PA"]
    for gid in gids:
        fake.move_task(gid, ["PA"])
    fake.delete_task("zdead")
    fake.add_task("LATER", projects=("PA",))
    return fake, state


@pytest.mark.parametrize(("movers", "budget", "runs_bound"), [(12, 10, 40), (1500, 1500, 8)])
def test_tasks_moved_in_bulk_settle_and_later_changes_still_arrive(movers, budget, runs_bound):
    """Each move is a live task the listing lacks, so each costs a check that answers 'still
    selected'. That answer is progress: a run that only made such checks used to repeat them
    for ever, forgetting nothing and ingesting nothing."""
    fake, state = _bulk_move(movers, budget)
    forgotten: set[str] = set()
    ingested: set[str] = set()

    for runs in range(1, runs_bound + 1):
        rows, stats = _sync(fake, state, max_requests=budget, project_gids=["PA", "PB"])
        forgotten |= _tombstones(rows)
        ingested |= set(_tasks(rows))
        if stats["failed"] == 0 and "listing" not in state:
            break

    assert stats["failed"] == 0 and runs < runs_bound, "the sync settles"
    assert forgotten == {"task:zdead"} and "task:LATER" in ingested
    assert "task:zdead" not in state["known_ids"] and "task:LATER" in state["known_ids"]


def test_a_task_that_left_the_selection_unread_counts_as_progress_too():
    """Fetched and then skipped (no selected project) is an answer as well: a kept listing whose
    tasks all left the selection must not be kept again and again."""
    fake = _three_projects(tasks_each=3)
    fake.add_project("P9", "Elsewhere")
    state: dict = {}
    _sync(fake, state, **SELECTION)
    for gid in ("11", "12", "13"):
        fake.edit_task(gid)
    fake.fail("/tasks", 429, after=1)
    _sync(fake, state, max_requests=100, **SELECTION)  # P1 lists, P2 is cut
    fake.heal()
    for gid in ("11", "12", "13"):
        fake.move_task(gid, ["P9"])  # P1's kept listing still names them
    forgotten: set[str] = set()

    for runs in range(1, 11):
        rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS * 2, **SELECTION)
        forgotten |= _tombstones(rows)
        if stats["failed"] == 0:
            break

    assert stats["failed"] == 0 and runs < 10, "the run that skips them does not repeat for ever"
    assert forgotten == {"task:11", "task:12", "task:13"}


# --- a check that cannot be answered ---


@pytest.mark.parametrize("failure", [503, None])
def test_a_check_answering_5xx_or_nothing_keeps_the_task_and_stops_nothing_else(failure):
    fake = _workspace(tasks=4)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", notes="Rewritten.")
    fake.delete_task("2")
    fake.delete_task("3")
    fake.fail("/tasks/2", failure)

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1"}, "the other tasks of the run still render"
    assert _tombstones(rows) == {"task:3"} and "task:2" in state["known_ids"]
    assert (stats["failed"], _reasons(stats)) == (1, set())
    fake.heal()

    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"task:2"} and stats["failed"] == 0


def test_three_checks_in_a_row_answering_5xx_stop_the_run_cleanly():
    """A hung Asana would spend the budget on checks as it would on reads."""
    fake = _workspace(tasks=6)
    state: dict = {}
    _sync(fake, state)
    for gid in "2345":
        fake.delete_task(gid)
        fake.fail(f"/tasks/{gid}", 500)
    fake.delete_task("6")

    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == set() and stats["failed_unavailable"] == 1
    assert stats["failed"] == 3, "the fourth check is never asked"
    assert {f"task:{n}" for n in range(2, 7)} <= set(state["known_ids"])
    # the three answers were progress, so the run committed instead of keeping the listing
    assert "listing" not in state and state["selection"] == ["P1"] and state["pending"] == []


def test_a_403_between_checks_answering_5xx_neither_counts_nor_resets_the_streak():
    fake = _workspace(tasks=6)
    state: dict = {}
    _sync(fake, state)
    for gid in "2345":
        fake.delete_task(gid)
    fake.fail("/tasks/2", 500)
    fake.fail("/tasks/3", 403)
    fake.fail("/tasks/4", 500)
    fake.fail("/tasks/5", 500)

    rows, stats = _sync(fake, state)

    assert stats["failed"] == 3 and stats["failed_unavailable"] == 1
    assert _tombstones(rows) == {"task:3"}, "the 403 is an answer: gone"


def test_the_checks_count_their_own_streak_apart_from_the_task_reads():
    """Two reads and two checks answering 5xx are four, but never three in a row of one kind."""
    fake = _workspace(tasks=5)
    state: dict = {}
    _sync(fake, state)
    for gid in "12":
        fake.edit_task(gid)
        fake.fail(f"/tasks/{gid}", 500)
    for gid in "45":
        fake.delete_task(gid)
        fake.fail(f"/tasks/{gid}", 500)

    _, stats = _sync(fake, state)

    assert stats["failed"] == 4 and _reasons(stats) == set()


def test_a_check_that_answers_resets_the_streak_of_checks_that_do_not():
    fake = FakeAsana()
    fake.add_project("P1")
    fake.add_project("P2")
    fake.add_task("a", projects=("P1",))
    for gid in "bcde":
        fake.add_task(gid, projects=("P2",))
    both = {"project_gids": ["P1", "P2"]}
    state: dict = {}
    _sync(fake, state, **both)
    original, listings = fake.respond, []

    def respond(path, params=None):
        answer = original(path, params)
        if path == "/tasks":
            listings.append(params["project"])
            if len(listings) == 1:  # P1 is listed: d moves there, b, c and e are deleted
                fake.move_task("d", ["P1"])
                for gid in "bce":
                    fake.delete_task(gid)
        return answer

    fake.respond = respond
    for gid in "bce":
        fake.fail(f"/tasks/{gid}", 500)

    _, stats = _sync(fake, state, **both)

    assert stats["failed"] == 3 and _reasons(stats) == set(), "b, c, then d answers, then e"
    assert _checked(fake) == ["b", "c", "d", "e"]


@pytest.mark.parametrize("data", [{}, {"memberships": None}, {"memberships": "P1"}])
def test_a_check_answer_without_a_memberships_list_raises_and_forgets_nothing(data):
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")
    fake.answers["/tasks/2"] = {"data": data}
    before = copy.deepcopy(state)

    with pytest.raises(AsanaResponseError, match="memberships"):
        list(_iter_rows(fake, state, new_stats(), project_gids=["P1"]))

    assert state == before


def test_an_empty_memberships_list_is_a_real_answer_the_task_is_in_no_project():
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")
    fake.answers["/tasks/2"] = {"data": {"memberships": []}}

    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == {"task:2"}


# --- what a cut run queues ---


def test_a_cut_in_the_check_phase_queues_no_task_that_was_rendered():
    fake = _workspace(tasks=3)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1")
    fake.edit_task("2")
    fake.delete_task("3")
    fake.fail("/tasks/3", 429)

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1", "task:2"} and stats["failed_rate_limit"] == 1
    assert state["pending"] == [] and stats["pending"] == 0


def test_a_queued_task_that_a_resolved_project_lists_is_read_and_leaves_the_queue():
    """Task x is in P1 and in the unresolved P2: P1 lists it, so it is read, not held."""
    fake = FakeAsana()
    fake.add_project("P1")
    fake.add_project("P2")
    fake.add_task("a", projects=("P1",))
    fake.add_task("x", projects=("P1", "P2"))
    both = {"project_gids": ["P1", "P2"]}
    state: dict = {}
    _sync(fake, state, **both)
    fake.edit_task("a")
    fake.edit_task("x")
    _sync(fake, state, max_requests=2 * LISTING_REQUESTS + TASK_REQUESTS, **both)
    assert state["pending"] == ["x"]
    fake.fail("/projects/P2", 403)

    rows, _ = _sync(fake, state, **both)

    assert "task:x" in _tasks(rows) and state["pending"] == []


def test_the_stored_progress_of_an_unresolved_project_is_not_touched_by_a_queue():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    state["retry"] = ["21"]  # a task of P2 sits in the retry queue
    stored = copy.deepcopy(state["projects"]["P2"])
    assert "21" in stored["seen"]
    fake.fail("/projects/P2", 403)

    _sync(fake, state, **SELECTION)

    assert state["projects"]["P2"] == stored and state["retry"] == ["21"]


# --- a rejected offset ---


def test_an_offset_asana_rejects_raises_on_every_run_and_leaves_the_state_alone():
    fake = _long_project(tasks=6, page_size=2)
    fake.fail("/tasks", 429, after=1)
    state: dict = {}
    _sync(fake, state)
    fake.heal()
    state["listing"]["partial"]["P1"]["offset"] = "never-issued"
    before = copy.deepcopy(state)

    for _ in range(2):
        with pytest.raises(AsanaAPIError) as raised:
            list(_iter_rows(fake, state, new_stats(), project_gids=["P1"]))
        assert raised.value.status == 400 and state == before


def test_an_offset_issued_for_another_listing_is_rejected_too():
    fake = _long_project(tasks=6, page_size=2)
    fake.add_project("P2")
    for n in range(1, 5):
        fake.add_task(f"2{n}", projects=("P2",))
    fake.page_size = 2
    fake.fail("/tasks", 429, after=1)
    state: dict = {}
    _sync(fake, state, project_gids=["P1", "P2"])
    fake.heal()
    foreign = fake.respond("/tasks", {"project": "P2", "opt_fields": "modified_at", "limit": 2})
    token = foreign.json()["next_page"]["offset"]
    state["listing"]["partial"]["P1"]["offset"] = token

    with pytest.raises(AsanaAPIError) as raised:
        list(_iter_rows(fake, state, new_stats(), project_gids=["P1", "P2"]))

    assert raised.value.status == 400


# --- a budget that cannot make progress raises ---


def _unresolved_workspace_one() -> tuple[FakeAsana, dict, dict]:
    """P0 will answer 403; P1 holds ``a`` (edited later) and ``b`` (deleted later)."""
    fake = FakeAsana()
    fake.add_project("P0")
    fake.add_project("P1")
    fake.add_task("u", projects=("P0",))
    fake.add_task("a", projects=("P1",))
    fake.add_task("b", projects=("P1",))
    kwargs = {"project_gids": ["P0", "P1"]}
    state: dict = {}
    _sync(fake, state, **kwargs)
    fake.fail("/projects/P0", 403)
    fake.edit_task("a", notes="Edited.")
    fake.delete_task("b")
    return fake, state, kwargs


def _unresolved_workspace_two() -> tuple[FakeAsana, dict, dict]:
    fake = FakeAsana()
    for gid in ("P0", "P0b", "P1"):
        fake.add_project(gid)
    fake.add_task("u", projects=("P0",))
    fake.add_task("v", projects=("P0b",))
    fake.add_task("a", projects=("P1",))
    kwargs = {"project_gids": ["P0", "P0b", "P1"]}
    state: dict = {}
    _sync(fake, state, **kwargs)
    fake.fail("/projects/P0", 403)
    fake.fail("/projects/P0b", 404)
    fake.edit_task("a", notes="Edited.")
    return fake, state, kwargs


def _big_comments_workspace() -> tuple[FakeAsana, dict, dict]:
    """Task ``big`` has 350 comments: its task read, four story pages and the subtasks are six
    requests, and ``z`` waits behind it."""
    fake = FakeAsana()
    fake.add_project("P1")
    for gid in ("a", "big", "z"):
        fake.add_task(gid)
    state: dict = {}
    _sync(fake, state)
    for n in range(350):
        fake.add_comment("big", "Grace", f"comment {n}")
    for gid in ("big", "z", "a"):
        fake.edit_task(gid, notes="Edited.")
    return fake, state, {}


STUCK = [
    ("one unresolved project", _unresolved_workspace_one, 3, 10),
    ("two unresolved projects", _unresolved_workspace_two, 4, 12),
    ("a task of four story pages", _big_comments_workspace, 5, 12),
]


@pytest.mark.parametrize(("name", "build", "small", "enough"), STUCK, ids=[c[0] for c in STUCK])
def test_a_budget_that_cannot_make_progress_raises_naming_max_requests(name, build, small, enough):
    """Every run would spend the same requests the same way and stop at the same place. It makes
    progress while it can (the listing is kept page by page); on the first run that cannot, it
    raises instead of looping, and the raise leaves the state as it was."""
    fake, state, kwargs = build()
    raised = None

    for _ in range(6):
        before = copy.deepcopy(state)
        try:
            _sync(fake, state, max_requests=small, **kwargs)
        except AsanaSourceError as exc:
            raised = exc
            break

    assert raised is not None, "a run that can only repeat itself must not end cleanly"
    assert f"max_requests={small}" in str(raised) and "Raise max_requests" in str(raised)
    assert state == before
    assert not isinstance(raised, AsanaAPIError)


@pytest.mark.parametrize(("name", "build", "small", "enough"), STUCK, ids=[c[0] for c in STUCK])
def test_the_same_selection_with_enough_budget_does_not_raise(name, build, small, enough):
    fake, state, kwargs = build()

    for _ in range(8):
        _, stats = _sync(fake, state, max_requests=enough, **kwargs)
        if stats["failed"] == 0:
            break

    assert stats["failed"] == 0


def test_a_rate_limit_cut_that_makes_no_progress_is_a_clean_stop_not_an_error():
    """A 429 is the provider's limit and clears with time, so unlike a spent budget it is never
    an error, however often it repeats."""
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1")
    fake.edit_task("2")
    fake.fail("/tasks/1", 429, times=1)
    _sync(fake, state)  # lists, then is cut at the first read: the listing is kept

    for _ in range(3):
        fake.fail("/tasks/1", 429, times=1)
        before = copy.deepcopy(state)
        rows, stats = _sync(fake, state)
        assert rows == [] and stats["failed_rate_limit"] == 1 and state == before


# --- an answer of the wrong shape is told, never read as a task that left the selection ---

_MISSING = object()
BAD_MEMBERSHIPS = {
    "a null entry": [None],
    "a gid that is a list": [{"project": {"gid": ["P1"]}}],
    "a gid that is a number": [{"project": {"gid": 5}}],
    "a project without a gid": [{"project": {"name": "Website"}}],
    "a project that is a string": [{"project": "P1"}],
    "a project that is a list": [{"project": ["P1"]}],
    "a good entry then a null entry": [
        {"project": {"gid": "P1", "name": "Website"}},
        None,
    ],
    "null": None,
    "missing": _MISSING,
}
GOOD_MEMBERSHIPS = [{"project": {"gid": "P1", "name": "Website launch"}, "section": None}]


def _body(**fields) -> dict:
    return {"data": {key: value for key, value in fields.items() if value is not _MISSING}}


PROJECTLESS_ENTRIES = {
    "no project key": {"section": {"name": "My Tasks"}},
    "a null project": {"project": None, "section": None},
}


@pytest.mark.parametrize("entry", PROJECTLESS_ENTRIES.values(), ids=PROJECTLESS_ENTRIES.keys())
def test_an_entry_without_a_project_is_skipped_and_the_task_is_still_selected(entry):
    """A task on someone's My Tasks list carries a membership that is not a project."""
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", notes="Edited.")
    fake.delete_task("2")
    fake.answers["/tasks/1"] = _body(
        gid="1", name="Task 1", notes="Edited.", memberships=[entry, *GOOD_MEMBERSHIPS, entry]
    )
    fake.answers["/tasks/2"] = _body(gid="2", memberships=[entry, *GOOD_MEMBERSHIPS])

    rows, stats = _sync(fake, state)

    task = _tasks(rows)["task:1"]
    assert "Project: Website launch" in task["content"] and task[NODE_SET_COLUMN] == ["asana:P1"]
    assert (stats["failed"], state["retry"]) == (0, []), "the render read is not an error"
    assert _tombstones(rows) == set() and "task:2" in state["known_ids"], "the check keeps task 2"


@pytest.mark.parametrize("entry", PROJECTLESS_ENTRIES.values(), ids=PROJECTLESS_ENTRIES.keys())
def test_a_task_with_only_projectless_entries_is_in_no_project(entry):
    """Like an empty list: a render read builds no document for it (it left the selection, no
    error, no retry) and a check forgets it."""
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", notes="Edited.")
    fake.delete_task("2")
    fake.answers["/tasks/1"] = _body(gid="1", name="Task 1", memberships=[entry])
    fake.answers["/tasks/2"] = _body(gid="2", memberships=[entry, entry])

    rows, stats = _sync(fake, state)

    assert "task:1" not in _tasks(rows) and (stats["failed"], state["retry"]) == (0, [])
    assert "task:1" in state["known_ids"]
    assert _tombstones(rows) == {"task:2"}


def test_a_projectless_entry_adds_no_project_line_and_no_node_set():
    task = {"memberships": [{"project": None}, {"section": {"name": "My Tasks"}}]}

    row = render_task("1", task, [], [], ["P1"])

    assert row["content"] == "Completed: no" and row[NODE_SET_COLUMN] == []


@pytest.mark.parametrize("shape", BAD_MEMBERSHIPS.values(), ids=BAD_MEMBERSHIPS.keys())
def test_a_render_read_with_bad_memberships_keeps_the_task_for_a_retry_and_never_raises(shape):
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", notes="Edited.")
    fake.answers["/tasks/1"] = _body(gid="1", name="Task 1", memberships=shape)

    rows, stats = _sync(fake, state)

    assert "task:1" not in _tasks(rows), "no document is built from it"
    assert (stats["failed"], state["retry"]) == (1, ["1"]), "told like any unreadable task"
    assert stats["kept"] == 1 and _reasons(stats) == set()


@pytest.mark.parametrize("shape", BAD_MEMBERSHIPS.values(), ids=BAD_MEMBERSHIPS.keys())
def test_a_check_with_bad_memberships_raises_and_forgets_nothing(shape):
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")
    fake.answers["/tasks/2"] = _body(gid="2", memberships=shape)
    before = copy.deepcopy(state)

    with pytest.raises(AsanaResponseError, match="memberships|membership"):
        list(_iter_rows(fake, state, new_stats(), project_gids=["P1"]))

    assert state == before


@pytest.mark.parametrize("fields", ["oops", ["oops"], [{"name": 5}], [{"display_value": ["x"]}]])
def test_a_render_read_with_bad_custom_fields_keeps_the_task_for_a_retry(fields):
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", notes="Edited.")
    fake.answers["/tasks/1"] = _body(
        gid="1", name="Task 1", memberships=GOOD_MEMBERSHIPS, custom_fields=fields
    )

    rows, stats = _sync(fake, state)

    assert "task:1" not in _tasks(rows)
    assert (stats["failed"], state["retry"]) == (1, ["1"])


def test_custom_fields_that_are_null_or_have_null_values_are_no_fields():
    row = render_task(
        "1",
        {
            "memberships": GOOD_MEMBERSHIPS,
            "custom_fields": [{"name": "Priority", "display_value": None}, {"name": None}],
        },
        [],
        [],
        ["P1"],
    )
    assert row["content"] == "Completed: no\nProject: Website launch"
    assert render_task("1", {"memberships": [], "custom_fields": None}, [], [], [])["content"]


def test_a_task_read_that_stays_malformed_never_makes_the_run_raise_across_runs():
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", notes="Edited.")
    fake.answers["/tasks/1"] = _body(gid="1", name="Task 1", memberships=[None])

    for _ in range(3):
        rows, stats = _sync(fake, state)
        assert stats["failed"] == 1 and state["retry"] == ["1"] and "task:1" not in _tasks(rows)
    del fake.answers["/tasks/1"]

    rows, stats = _sync(fake, state)

    assert "task:1" in _tasks(rows) and state["retry"] == [] and stats["failed"] == 0


# --- inside a project: the page offset ---


def _long_project(tasks: int = 7, page_size: int = 2) -> FakeAsana:
    fake = FakeAsana(page_size=page_size)
    fake.add_project("P1")
    for n in range(1, tasks + 1):
        fake.add_task(str(n), f"Task {n}", projects=("P1",))
    return fake


def _task_pages(fake: FakeAsana) -> list[str | None]:
    """The offset of every request of the project's task listing, in order."""
    return [params.get("offset") for path, params in fake.calls if path == "/tasks"]


def test_a_429_inside_a_project_resumes_at_the_page_it_stopped_at():
    fake = _long_project()
    fake.fail("/tasks", 429, after=2)  # pages 1 and 2 are served, page 3 is cut
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _tasks(rows) == {} and stats["failed_rate_limit"] == 1
    progress = state["listing"]["partial"]["P1"]
    assert fake.offset_position(progress["offset"]) == 4
    assert sorted(progress["listed"]) == ["1", "2", "3", "4"]
    assert state["listing"]["done"] == {}
    fake.heal()
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    pages = _task_pages(fake)
    assert [fake.offset_position(token) for token in pages] == [4, 6], "pages 1, 2 not read again"
    assert set(_tasks(rows)) == {f"task:{n}" for n in range(1, 8)}
    assert stats["scanned"] == 7 and "listing" not in state


def test_the_offset_is_cleared_when_the_project_finishes_and_holds_no_text():
    fake = _long_project()
    fake.fail("/tasks", 429, after=1)
    state: dict = {}
    _sync(fake, state)
    kept = json.dumps(state["listing"])

    assert "Task" not in kept and "Project" not in kept
    fake.heal()
    _sync(fake, state, max_requests=LISTING_REQUESTS + 2)

    # the project finished: its progress moved into the done entry and the offset with it
    assert state["listing"]["partial"] == {} and sorted(state["listing"]["done"]) == ["P1"]


def test_a_changed_selection_drops_the_offset_of_a_cut_project():
    fake = _long_project()
    fake.add_project("P2")
    fake.fail("/tasks", 429, after=1)
    state: dict = {}
    _sync(fake, state)
    assert fake.offset_position(state["listing"]["partial"]["P1"]["offset"]) == 2
    fake.heal()
    fake.calls.clear()

    _sync(fake, state, project_gids=["P1", "P2"])

    assert _task_pages(fake)[0] is None, "the new pass starts P1 at its first page"


def test_a_project_with_more_pages_than_fit_between_two_429s_completes_over_runs(tmp_path):
    """Four requests a window: the project read, the events and two pages, so a project of three
    pages could never be listed if every run began at its first page."""
    fake = _long_project(tasks=3, page_size=1)
    fake.rate_limit(4)
    staged: dict[str, str] = {}
    first_offsets: list[str | None] = []

    for _ in range(30):
        fake.new_window()
        fake.calls.clear()
        source = _source(fake, project_gids=["P1"])
        staged = _run(tmp_path, source, nothing_staged_yet=True)
        first_offsets.append(next(iter(_task_pages(fake)), "none"))
        if source.cognee_sync_stats["failed"] == 0:
            break

    assert source.cognee_sync_stats["failed"] == 0
    assert set(staged) == {"project:P1", "task:1", "task:2", "task:3"}
    assert first_offsets[0] is None and fake.offset_position(first_offsets[1]) == 2


@pytest.mark.parametrize(
    ("change", "rendered"),
    [
        ("comment", {"task:2"}),
        ("rename", {"task:1", "task:2", "task:3"}),
        ("token", {"task:1", "task:2", "task:3"}),
    ],
)
def test_what_the_events_said_before_a_cut_survives_the_resume(change, rendered):
    """The run reads the feed and is cut on the first task page. The feed is consumed: only what
    was kept can tell the resuming run which tasks changed (a comment, a rename, a feed gap)."""
    fake = _workspace(tasks=3)
    state: dict = {}
    _sync(fake, state)
    if change == "comment":
        fake.add_comment("2", "Grace", "new comment")
    elif change == "rename":
        fake.rename("project", "P1", "Relaunch")
    else:
        fake.expire_token("P1")
    fake.fail("/tasks", 429, after=0, times=1)

    rows, stats = _sync(fake, state)

    assert _tasks(rows) == {} and stats["failed_rate_limit"] == 1
    kept = state["listing"]["partial"]["P1"]
    assert (
        kept["polled"]
        and (kept["events"], kept["renamed"], kept["gap"])
        == {
            "comment": (["2"], False, False),
            "rename": ([], True, False),
            "token": ([], False, True),
        }[change]
    )

    rows, _ = _sync(fake, state)

    assert set(_tasks(rows)) == rendered


def test_an_unresolved_project_loses_its_progress_inside_the_pass_and_starts_over():
    fake = _long_project(tasks=6, page_size=2)
    fake.add_project("P2")
    fake.fail("/tasks", 429, after=1)  # P1's second page is cut
    state: dict = {}
    _sync(fake, state, project_gids=["P1", "P2"])
    assert "P1" in state["listing"]["partial"]
    fake.heal()
    fake.fail("/projects/P1", 403)
    fake.fail("/tasks", 429)  # P2 is cut on its first page

    _sync(fake, state, project_gids=["P1", "P2"])

    assert "P1" not in state["listing"]["partial"] and "P2" in state["listing"]["partial"]
    fake.heal()
    fake.calls.clear()

    _sync(fake, state, project_gids=["P1", "P2"])

    assert _task_pages(fake)[0] is None, "P1 starts at its first page when it returns"


def test_a_task_that_left_the_selection_before_it_was_read_builds_no_document():
    fake = _workspace(tasks=2)
    fake.add_project("P9", "Elsewhere")
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2", notes="Changed.")
    original, listed = fake.respond, []

    def respond(path, params=None):
        answer = original(path, params)
        if path == "/tasks":
            listed.append(path)
            fake.move_task("2", ["P9"])  # after the listing, before the read
        return answer

    fake.respond = respond

    rows, _ = _sync(fake, state)

    assert _tasks(rows) == {} and "task:2" in state["known_ids"]
    fake.respond = original
    rows, _ = _sync(fake, state)
    assert _tombstones(rows) == {"task:2"}


def test_a_pass_cut_before_its_first_task_keeps_the_pending_of_the_last_one_and_reports_none():
    """``pending`` counts what this run queued: a run kept before it rendered anything queued
    nothing, and the state still holds what the run before it left."""
    fake = _workspace(tasks=3)
    state: dict = {}
    _sync(fake, state)
    for gid in "123":
        fake.edit_task(gid)
    _sync(fake, state, max_requests=LISTING_REQUESTS + TASK_REQUESTS)
    assert state["pending"] == ["2", "3"]

    rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS)

    assert _tasks(rows) == {} and stats["failed_budget"] == 1
    assert stats["pending"] == 0 and state["pending"] == ["2", "3"]


# --- a project Asana will not show ---


def test_a_403_and_a_404_project_do_not_stop_the_other_projects():
    fake = _three_projects()
    fake.fail("/projects/P1", 403)
    fake.fail("/projects/P2", 404)
    state: dict = {}

    rows, stats = _sync(fake, state, **SELECTION)

    assert [row["id"] for row in rows if row["id"].startswith("project:")] == ["project:P3"]
    assert set(_tasks(rows)) == {"task:31", "task:32"}
    assert (stats["unresolved"], stats["failed"], stats["deleted"]) == (2, 0, 0)
    assert sorted(state["projects"]) == ["P3"], "an unresolved project has no stored progress yet"
    assert state["known_ids"] == ["project:P3", "task:31", "task:32"], "no row never emitted"


def test_an_unresolved_project_keeps_its_row_and_its_tasks_across_runs():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    stored = copy.deepcopy(state["projects"]["P2"])
    fake.fail("/projects/P2", 403)

    for run in range(3):
        fake.edit_task("11", notes=f"Rewritten {run}.")
        if run == 0:
            fake.delete_task("32")
        rows, stats = _sync(fake, state, **SELECTION)
        assert not {"project:P2", "task:21", "task:22"} & _tombstones(rows)
        assert stats["unresolved"] == 1 and "task:11" in _tasks(rows)
        assert {"project:P2", "task:21", "task:22"} <= set(state["known_ids"])
        assert state["projects"]["P2"] == stored, "cursor, token, seen and tasks are untouched"

    assert "task:32" not in state["known_ids"], "a task of a resolved project is still forgotten"


def test_the_tasks_of_an_unresolved_project_are_protected_and_never_checked():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.fail("/projects/P2", 403)
    fake.delete_task("12")
    fake.calls.clear()

    rows, _ = _sync(fake, state, **SELECTION)

    assert _checked(fake) == ["12"], "task 12 is missing and asked for; P2's tasks are not"
    assert _tombstones(rows) == {"task:12"}
    assert {"project:P2", "task:21", "task:22"} <= set(state["known_ids"])
    assert "/projects/P2" in [path for path, _ in fake.calls], "it is asked again every run"


def test_a_task_kept_for_a_retry_stays_queued_while_its_project_is_unresolved():
    """``kept`` is the retry queue the run committed: the task carried over from the unresolved
    project counts although this run did not queue it itself."""
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2")
    fake.fail("/tasks/2", 403)
    _sync(fake, state)
    assert state["retry"] == ["2"]
    fake.heal()
    fake.fail("/projects/P1", 403)

    _, stats = _sync(fake, state)

    assert stats["unresolved"] == 1 and state["retry"] == ["2"] and stats["kept"] == 1
    fake.heal()

    rows, _ = _sync(fake, state)

    assert "task:2" in _tasks(rows) and state["retry"] == []


def test_each_cause_of_an_unresolved_project_gets_its_own_warning_without_a_gid(caplog):
    fake = _three_projects()
    fake.fail("/projects/P1", 403)
    fake.fail("/projects/P2", 404)

    _sync(fake, {}, **SELECTION)

    warnings = [r for r in caplog.messages if "answered" in r]
    assert any("1 project(s) answered 403" in m for m in warnings)
    assert any("1 project(s) answered 404" in m for m in warnings)
    assert not any("P1" in m or "P2" in m for m in warnings)


def test_an_unresolved_project_syncs_again_when_access_returns():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.fail("/projects/P2", 404)
    fake.edit_task("21", notes="Edited meanwhile.")
    _sync(fake, state, **SELECTION)
    fake.heal()

    rows, stats = _sync(fake, state, **SELECTION)

    assert stats["unresolved"] == 0
    assert "project:P2" in [row["id"] for row in rows]
    assert "Edited meanwhile." in _tasks(rows)["task:21"]["content"]
    assert set(_tasks(rows)) == {"task:21"}, "only what changed while it was out of reach"


def test_a_401_on_a_project_still_raises():
    fake = _three_projects()
    fake.fail("/projects/P2", 401)
    stats = new_stats()

    with pytest.raises(AsanaAuthError):
        list(_iter_rows(fake, {}, stats, project_gids=["P1", "P2", "P3"]))

    assert stats["unresolved"] == 0 and stats["failed_auth"] == 1


def test_deselecting_an_unresolved_project_forgets_it_and_its_tasks():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    fake.fail("/projects/P2", 403)
    _sync(fake, state, **SELECTION)

    rows, stats = _sync(fake, state, project_gids=["P1", "P3"])

    assert _tombstones(rows) == {"project:P2", "task:21", "task:22"}
    assert stats["unresolved"] == 0 and sorted(state["projects"]) == ["P1", "P3"]


def test_a_run_where_every_project_is_unresolved_ends_cleanly_and_forgets_nothing():
    fake = _three_projects()
    state: dict = {}
    _sync(fake, state, **SELECTION)
    known = list(state["known_ids"])
    fake.fail("/projects", 403)

    rows, stats = _sync(fake, state, **SELECTION)

    assert rows == [] and stats["unresolved"] == 3 and stats["failed"] == 0
    assert state["known_ids"] == known
    assert _reasons(stats) == set() and _checked(fake) == []


def test_a_first_run_where_every_project_is_unresolved_emits_nothing():
    fake = _three_projects()
    fake.fail("/projects", 404)
    state: dict = {}

    rows, stats = _sync(fake, state, **SELECTION)

    assert rows == [] and stats["unresolved"] == 3 and stats["scanned"] == 0
    assert state["known_ids"] == [], "a project row that was never emitted is not known"


def test_an_unresolved_project_resolves_through_a_real_dlt_run(tmp_path):
    fake = _three_projects()
    _run(tmp_path, _source(fake, **SELECTION))
    fake.fail("/projects/P2", 403)
    fake.edit_task("11", notes="Rewritten.")

    source = _source(fake, **SELECTION)
    staged = _run(tmp_path, source)

    assert source.cognee_sync_stats["unresolved"] == 1
    assert {"project:P2", "task:21", "task:22"} <= set(staged), "nothing of it is forgotten"
    assert "Rewritten." in staged["task:11"]


# --- forgetting ---


def test_a_task_missing_from_a_complete_listing_is_tombstoned():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")

    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"task:2"} and stats["deleted"] == 1
    assert state["known_ids"] == ["project:P1", "task:1", "task:3"]


@pytest.mark.parametrize(
    ("prefix", "status"),
    [("/tasks", 500), ("/events", 403), ("/events", 404), ("/projects/P1", 500)],
)
def test_a_listing_that_fails_forgets_nothing_and_moves_no_state(prefix, status):
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")
    before = copy.deepcopy(state)
    fake.fail(prefix, status)

    with pytest.raises(AsanaAPIError) as raised:
        stats = new_stats()
        list(_iter_rows(fake, state, stats, project_gids=["P1"]))

    assert raised.value.status == status
    assert state == before


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"data": "x"},
        {"data": {"gid": "1"}},
        {"data": [1]},
        {"data": [{"modified_at": "2026-10-01T10:00:00.000Z"}]},
        {"data": [{"gid": "../projects", "modified_at": "2026-10-01T10:00:00.000Z"}]},
        {"data": [{"gid": "1"}]},
        {"data": [{"gid": "1", "modified_at": None}]},
        {"data": [{"gid": "1", "modified_at": "yesterday"}]},
        {"data": [{"gid": "1", "modified_at": ""}]},
        {"data": [{"gid": "1", "modified_at": 20261001}]},
        {"data": [{"gid": "1", "modified_at": "2026-10-01T10:00:00"}]},  # no offset: no instant
        {"data": [{"gid": "1", "modified_at": "0001-01-01T00:00:00+02:00"}]},  # before UTC's year 1
        {"data": [], "next_page": {}},
        {"data": [], "next_page": "x"},
        {"data": [], "next_page": {"offset": ""}},
    ],
)
def test_a_listing_answer_of_the_wrong_shape_raises_and_is_not_read_as_empty(answer):
    """A 200 that says nothing useful must not look like a project with no tasks: every
    synced task would then be tombstoned."""
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    before = copy.deepcopy(state)
    fake.answers["/tasks"] = answer
    stats = new_stats()

    with pytest.raises(AsanaResponseError):
        list(_iter_rows(fake, state, stats, project_gids=["P1"]))

    assert state == before
    assert (stats["failed"], stats["deleted"]) == (1, 0)


def test_a_repeated_pagination_offset_raises_instead_of_looping():
    fake = _workspace(tasks=5)
    fake.page_size = 2
    fake.stuck_offset = "stuck"

    with pytest.raises(AsanaResponseError, match="offset came back twice"):
        _sync(fake, {})


def _malformed_task_workspace(path: str, answer) -> tuple[FakeAsana, dict, dict]:
    """Three synced tasks, all edited, with ``answer`` (a 200) served for ``path`` of task 2."""
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    before = copy.deepcopy(state)
    fake.edit_task("1")
    fake.edit_task("2", notes="Rewritten.")
    fake.edit_task("3")
    fake.answers[path] = answer
    return fake, state, before


def _assert_task_2_kept_and_recovers(fake: FakeAsana, state: dict, before: dict, path: str):
    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1", "task:3"}, "the other tasks of the run are rendered"
    assert _tombstones(rows) == set() and stats["deleted"] == 0, "the old document stays"
    assert "task:2" in state["known_ids"]
    assert (stats["failed"], stats["kept"]) == (1, 1)
    assert _reasons(stats) == set(), "a malformed body is no outage and no stopped run"
    assert state["retry"] == ["2"] and state["pending"] == []
    assert state["projects"]["P1"]["cursor"] > before["projects"]["P1"]["cursor"], "committed"

    del fake.answers[path]
    rows, stats = _sync(fake, state)

    assert "Rewritten." in _tasks(rows)["task:2"]["content"]
    assert (stats["failed"], stats["kept"], state["retry"]) == (0, 0, [])


@pytest.mark.parametrize("answer", [{}, {"data": []}, {"data": "x"}])
def test_a_task_answer_of_the_wrong_shape_keeps_the_task_and_the_run_goes_on(answer):
    fake, state, before = _malformed_task_workspace("/tasks/2", answer)

    _assert_task_2_kept_and_recovers(fake, state, before, "/tasks/2")


@pytest.mark.parametrize("answer", [{"data": [1, 2]}, {"data": "x"}, {}])
def test_a_comment_listing_of_the_wrong_shape_keeps_the_task_and_the_run_goes_on(answer):
    fake, state, before = _malformed_task_workspace("/tasks/2/stories", answer)

    _assert_task_2_kept_and_recovers(fake, state, before, "/tasks/2/stories")


@pytest.mark.parametrize("answer", [{"data": [1]}, {"data": [{"gid": "s"}, "x"]}, {"data": None}])
def test_a_subtask_listing_of_the_wrong_shape_keeps_the_task_and_the_run_goes_on(answer):
    fake, state, before = _malformed_task_workspace("/tasks/2/subtasks", answer)

    _assert_task_2_kept_and_recovers(fake, state, before, "/tasks/2/subtasks")


@pytest.mark.parametrize("listing", ["stories", "subtasks"])
@pytest.mark.parametrize(
    "next_page",
    [{}, {"offset": ""}, {"offset": 5}, "x"],
    ids=["no offset", "empty offset", "offset not a string", "not an object"],
)
def test_a_next_page_without_a_usable_offset_inside_a_task_keeps_the_task(listing, next_page):
    path = f"/tasks/2/{listing}"
    fake, state, before = _malformed_task_workspace(path, {"data": [], "next_page": next_page})

    _assert_task_2_kept_and_recovers(fake, state, before, path)


@pytest.mark.parametrize("listing", ["stories", "subtasks"])
def test_a_repeated_pagination_offset_inside_a_task_keeps_the_task_instead_of_looping(listing):
    path = f"/tasks/2/{listing}"
    # The same page, and so the same next offset, comes back every time it is asked for.
    fake, state, before = _malformed_task_workspace(
        path, {"data": [], "next_page": {"offset": "again"}}
    )

    _assert_task_2_kept_and_recovers(fake, state, before, path)


def test_a_malformed_task_never_stops_the_run_not_even_three_in_a_row_across_runs():
    fake = _workspace(tasks=6)
    fake.answers["/tasks/2"] = {"data": "x"}
    fake.answers["/tasks/3/stories"] = {"data": [1]}
    fake.answers["/tasks/4/subtasks"] = {"data": [1]}
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1", "task:5", "task:6"}, "the tasks after them are read"
    assert (stats["failed"], stats["kept"], stats["scanned"]) == (3, 3, 6)
    assert _reasons(stats) == set(), "three malformed bodies in a row are no outage"
    assert state["retry"] == ["2", "3", "4"] and state["pending"] == []
    assert state["known_ids"] == ["project:P1", "task:1", "task:5", "task:6"], "never ingested"

    for run in range(3):
        fake.add_task(f"9{run}")
        fake.edit_task("5", notes=f"Edit {run}.")
        fake.calls.clear()
        rows, stats = _sync(fake, state)

        assert set(_tasks(rows)) == {"task:5", f"task:9{run}"}, "fresh tasks never wait for them"
        assert _fetched(fake) == ["5", f"9{run}", "2", "3", "4"], "the retries go last, once each"
        assert (stats["failed"], _reasons(stats)) == (3, set()), "no outage in the retry tail"
        assert state["retry"] == ["2", "3", "4"]

    fake.answers.clear()
    rows, stats = _sync(fake, state)

    assert {"task:2", "task:3", "task:4"} <= set(_tasks(rows)) and state["retry"] == []
    assert (stats["failed"], stats["kept"]) == (0, 0)


def test_a_malformed_task_neither_counts_nor_resets_the_streak_of_tasks_answering_5xx():
    fake = _workspace(tasks=6)
    fake.fail("/tasks/1", 500)
    fake.fail("/tasks/2", 500)
    fake.answers["/tasks/3"] = {"data": "x"}
    fake.fail("/tasks/4", 500)
    state: dict = {}

    _, stats = _sync(fake, state)

    assert (stats["failed"], _reasons(stats)) == (4, {"failed_unavailable"})
    assert _fetched(fake) == ["1", "2", "3", "4"], "the third 5xx in a row stops the run"


def test_a_project_listing_answer_of_the_wrong_shape_still_raises_though_a_task_answer_is_kept():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2")
    fake.answers["/tasks/2"] = {"data": "x"}
    _sync(fake, state)
    before = copy.deepcopy(state)
    fake.answers["/tasks"] = {"data": "x"}

    with pytest.raises(AsanaResponseError):
        _sync(fake, state)

    assert state == before and state["retry"] == ["2"], "the kept task is still queued"


@pytest.mark.parametrize("answer", [{}, {"data": []}, {"data": "x"}])
def test_a_check_answer_of_the_wrong_shape_raises_and_forgets_nothing(answer):
    """Unlike a task read, a check decides about forgetting: it raises, it never guesses."""
    fake = _workspace(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")
    fake.answers["/tasks/2"] = answer
    before = copy.deepcopy(state)

    with pytest.raises(AsanaResponseError):
        _sync(fake, state)

    assert state == before


# --- a task that cannot be read ---


@pytest.mark.parametrize("status", [403, 404, 500, None])
def test_a_listed_task_that_cannot_be_read_keeps_its_document_and_is_asked_again(status):
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2", notes="Rewritten.")
    fake.fail("/tasks/2", status)

    rows, stats = _sync(fake, state)

    assert _tasks(rows) == {} and _tombstones(rows) == set()
    assert (stats["failed"], stats["kept"], stats["deleted"]) == (1, 1, 0)
    assert _reasons(stats) == set(), "a task that fails is not a stopped run"
    assert state["retry"] == ["2"] and "task:2" in state["known_ids"]
    fake.heal()
    rows, stats = _sync(fake, state)
    assert "Rewritten." in _tasks(rows)["task:2"]["content"]
    assert (stats["failed"], stats["kept"], state["retry"]) == (0, 0, [])


def test_three_unreadable_tasks_in_a_row_never_stop_the_run_nor_block_new_tasks():
    fake = _workspace(tasks=6)
    for gid, status in (("2", 403), ("3", 404), ("4", 403)):
        fake.fail(f"/tasks/{gid}", status)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1", "task:5", "task:6"}
    assert (stats["failed"], stats["kept"], stats["scanned"]) == (3, 3, 6)
    assert _reasons(stats) == set()
    assert state["retry"] == ["2", "3", "4"]
    assert state["known_ids"] == ["project:P1", "task:1", "task:5", "task:6"], "never ingested"

    fake.add_task("7")
    fake.edit_task("5", notes="Edited.")
    fake.calls.clear()
    rows, stats = _sync(fake, state)

    # fresh tasks first, the three retries last, each asked once
    assert _fetched(fake) == ["5", "7", "2", "3", "4"]
    assert set(_tasks(rows)) == {"task:5", "task:7"}
    assert (stats["failed"], _reasons(stats)) == (3, set())


def test_a_task_that_stays_unreadable_for_many_runs_is_never_forgotten_and_asked_once_a_run():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2", notes="Rewritten.")
    fake.fail("/tasks/2", 403)

    for _ in range(6):
        fake.calls.clear()
        rows, stats = _sync(fake, state)
        assert _tombstones(rows) == set() and stats["deleted"] == 0
        assert len(fake.requested("/tasks/2")) == 1, "a retry task fails at most once per run"

    assert "task:2" in state["known_ids"]


def test_a_retry_task_that_leaves_the_listing_is_forgotten_and_dropped():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2")
    fake.fail("/tasks/2", 404)
    _sync(fake, state)
    assert state["retry"] == ["2"]
    fake.delete_task("2")

    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == {"task:2"}
    assert state["retry"] == [] and "task:2" not in state["known_ids"]


def test_retries_go_last_and_a_permanent_403_or_404_never_counts_toward_an_outage():
    fake = _workspace(tasks=6)
    for gid in "23":
        fake.fail(f"/tasks/{gid}", 403)
    for gid in "45":
        fake.fail(f"/tasks/{gid}", 404)
    state: dict = {}
    _sync(fake, state)
    fake.add_task("9")
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert _fetched(fake)[0] == "9", "a fresh task is rendered before every retry"
    assert set(_tasks(rows)) == {"task:9"}
    assert (stats["failed"], _reasons(stats)) == (4, set()), "four retries are no outage"
    assert state["retry"] == ["2", "3", "4", "5"] and state["pending"] == []


def _retry_workspace(tasks: int = 5) -> tuple[FakeAsana, dict]:
    """Every task but the first kept for a retry (403 once), the fake then healed."""
    fake = _workspace(tasks=tasks)
    for gid in range(2, tasks + 1):
        fake.fail(f"/tasks/{gid}", 403)
    state: dict = {}
    _sync(fake, state)
    fake.heal()
    return fake, state


@pytest.mark.parametrize("failure", [500, None])
def test_three_retries_in_a_row_answering_5xx_stop_the_run_and_the_next_one_resumes(failure):
    """A hung Asana would otherwise cost every retry its full retry backoff with the staging lock
    held."""
    fake, state = _retry_workspace()
    for gid in "2345":
        fake.fail(f"/tasks/{gid}", failure)
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert _fetched(fake) == ["2", "3", "4"], "the fourth retry is never asked"
    assert _tasks(rows) == {}
    assert (stats["failed"], stats["failed_unavailable"], stats["deleted"]) == (3, 1, 0)
    assert state["retry"] == ["2", "3", "4", "5"] and state["pending"] == []
    assert (stats["kept"], stats["pending"]) == (4, 0)
    fake.heal()

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {f"task:{gid}" for gid in "2345"}
    assert (stats["failed"], _reasons(stats), state["retry"]) == (0, set(), [])


def test_a_success_between_retries_answering_5xx_resets_the_tail_streak():
    fake, state = _retry_workspace(tasks=6)
    for gid in "2356":
        fake.fail(f"/tasks/{gid}", 500)

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:4"}
    assert (stats["failed"], _reasons(stats)) == (4, set())


def test_a_403_between_retries_answering_5xx_neither_counts_nor_resets_the_tail_streak():
    fake, state = _retry_workspace(tasks=6)
    fake.fail("/tasks/2", 500)
    fake.fail("/tasks/3", 403)
    fake.fail("/tasks/4", 500)
    fake.fail("/tasks/5", 500)

    _, stats = _sync(fake, state)

    assert (stats["failed"], stats["failed_unavailable"]) == (4, 1)


def test_the_tail_counts_its_5xx_apart_from_the_fresh_tasks():
    """Two failing fresh tasks and two failing retries are four, but not three in a row of one
    kind: neither count reaches the threshold."""
    fake, state = _retry_workspace(tasks=3)
    fake.add_task("8")
    fake.add_task("9")
    for gid in "8923":
        fake.fail(f"/tasks/{gid}", 500)

    rows, stats = _sync(fake, state)

    assert _tasks(rows) == {}
    assert (stats["failed"], _reasons(stats)) == (4, set())
    assert state["retry"] == ["2", "3", "8", "9"]


@pytest.mark.parametrize("failure", [500, None])
def test_three_fresh_failures_in_a_row_stop_the_run_cleanly_and_the_next_one_resumes(failure):
    fake = _workspace(tasks=6)
    for gid in "234":
        fake.fail(f"/tasks/{gid}", failure)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1"}
    assert (stats["failed"], stats["failed_unavailable"], stats["deleted"]) == (3, 1, 0)
    assert (stats["kept"], stats["pending"]) == (3, 2)
    assert state["retry"] == ["2", "3", "4"] and state["pending"] == ["5", "6"]
    fake.heal()
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert _fetched(fake) == ["5", "6", "2", "3", "4"], "what was not reached first"
    assert set(_tasks(rows)) == {f"task:{gid}" for gid in "23456"}
    assert (stats["failed"], stats["kept"], stats["pending"]) == (0, 0, 0)


def test_a_success_between_server_errors_resets_the_outage_streak():
    fake = _workspace(tasks=6)
    for gid in "246":
        fake.fail(f"/tasks/{gid}", 500)

    rows, stats = _sync(fake, {})

    assert set(_tasks(rows)) == {"task:1", "task:3", "task:5"}
    assert (stats["failed"], _reasons(stats)) == (3, set())


def test_a_task_that_always_answers_500_with_a_retry_after_never_stops_the_others(monkeypatch):
    fake = _workspace(tasks=4)
    fake.fail("/tasks/3", 500, headers={"Retry-After": "3600"})
    sleeps: list[float] = []
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=sleeps.append)
    state: dict = {}

    for run in range(1, 4):
        fake.add_task(f"1{run}")
        rows, stats = _sync(client, state)
        assert f"task:1{run}" in _tasks(rows)
        assert (stats["failed"], _reasons(stats)) == (1, set())

    assert sleeps == [1, 2] * 3, "Retry-After is not obeyed: only the short fixed backoff"
    assert "task:3" not in state["known_ids"] and state["retry"] == ["3"]


# --- a run that stops ---


def test_a_rejected_token_with_nothing_emitted_raises_and_moves_nothing():
    fake = _workspace()
    fake.fail("/projects", 401)
    state: dict = {}
    stats = new_stats()

    with pytest.raises(AsanaAuthError) as raised:
        list(_iter_rows(fake, state, stats, project_gids=["P1"]))

    assert str(raised.value) == "Asana /projects/{gid}: HTTP 401"
    assert state == {}
    assert (stats["failed"], stats["failed_auth"]) == (1, 1)


def test_a_rejected_token_after_rows_stops_cleanly_and_queues_the_rest():
    fake = _workspace(tasks=5)
    fake.fail("/tasks/3", 401)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert [row["id"] for row in rows] == ["project:P1", "task:1", "task:2"]
    assert (stats["failed"], stats["failed_auth"], stats["deleted"]) == (1, 1, 0)
    assert state["pending"] == ["3", "4", "5"] and state["retry"] == []
    assert (stats["pending"], stats["kept"]) == (3, 0)
    assert state["known_ids"] == ["project:P1", "task:1", "task:2"]
    assert state["projects"]["P1"]["cursor"] == fake.tasks["5"]["modified_at"], "the cursor moved"
    fake.heal()
    rows, stats = _sync(fake, state)
    assert set(_tasks(rows)) == {"task:3", "task:4", "task:5"}
    assert stats["failed"] == 0 and state["pending"] == []


def test_a_pending_task_older_than_the_overlap_window_is_still_read_next_run():
    """The cursor moves to the newest task on a stop, and only tasks near it are re-checked by
    their timestamp: an edited task an hour older than the newest is found through ``pending``."""
    fake = _workspace(tasks=4)
    state: dict = {}
    _sync(fake, state)
    for hour, gid in enumerate("1234", start=1):
        fake.tasks[gid].update(
            notes=f"Edit of {gid}.", modified_at=f"2026-11-01T0{hour}:00:00.000Z"
        )
    fake.fail("/tasks/2", 401)

    rows, _ = _sync(fake, state)

    assert set(_tasks(rows)) == {"task:1"} and state["pending"] == ["2", "3", "4"]
    fake.heal()
    rows, stats = _sync(fake, state)
    assert set(_tasks(rows)) == {"task:2", "task:3", "task:4"}
    assert state["pending"] == [] and stats["failed"] == 0


def test_a_rate_limit_while_listing_stops_cleanly_with_nothing_moved():
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("2")
    before = copy.deepcopy(state)
    fake.fail("/tasks", 429)

    rows, stats = _sync(fake, state)

    # the events the run read are kept for the page it was cut on; nothing else moved
    kept = state.pop("listing")
    assert rows == [] and state == before
    assert kept["done"] == {} and kept["partial"]["P1"]["polled"] is True
    assert (stats["failed"], stats["failed_rate_limit"], stats["deleted"]) == (1, 1, 0)


# Whatever Retry-After says, a 429 stops the run at once with nothing slept: a wait would hold
# the staging lock, and a negative or NaN value used to crash the old retry loop.
@pytest.mark.parametrize("where", ["/tasks", "/tasks/2/stories"])
@pytest.mark.parametrize(
    "retry_after",
    ["-5", "nan", "-inf", "inf", "1e308", "0", "soon", "Wed, 21 Oct 2026 07:28:00 GMT"],
)
def test_a_rate_limit_stops_cleanly_without_sleeping_whatever_retry_after_says(
    monkeypatch, where, retry_after
):
    fake = _workspace(tasks=3)
    fake.fail(where, 429, headers={"Retry-After": retry_after})
    sleeps: list[float] = []
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=sleeps.append)
    state: dict = {}

    rows, stats = _sync(client, state)

    assert sleeps == []
    assert (stats["failed"], stats["failed_rate_limit"], stats["deleted"]) == (1, 1, 0)
    if where == "/tasks/2/stories":
        assert _tasks(rows).keys() == {"task:1"}
        assert state["pending"] == ["2", "3"]
    else:
        # only the listing's progress is kept, never a task or a forgotten id
        assert rows == [] and set(state) == {"v", "known_ids", "listing"}
        assert state["listing"]["done"] == {}


def test_a_spent_budget_stops_cleanly_and_the_next_run_finishes():
    fake = _workspace(tasks=4)
    state: dict = {}
    budget = LISTING_REQUESTS + TASK_REQUESTS

    rows, stats = _sync(fake, state, max_requests=budget)

    assert set(_tasks(rows)) == {"task:1"}
    # the budget counts the listing's requests and the reads' alike
    assert (stats["failed"], stats["failed_budget"]) == (1, 1)
    assert stats["requests"] == budget
    assert state["pending"] == ["2", "3", "4"]
    rows, stats = _sync(fake, state)
    assert set(_tasks(rows)) == {"task:2", "task:3", "task:4"} and stats["failed"] == 0


def test_a_budget_spent_in_the_middle_of_a_task_leaves_that_task_pending():
    fake = _workspace(tasks=2)
    state: dict = {}

    # one task and the task read of the next: its comments are the request the budget cannot pay
    rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS + TASK_REQUESTS + 1)

    assert set(_tasks(rows)) == {"task:1"} and stats["failed_budget"] == 1
    assert state["pending"] == ["2"] and state["known_ids"] == ["project:P1", "task:1"]


def test_a_run_that_spends_its_whole_budget_inside_the_listing_keeps_its_progress():
    """The listing counts against the budget, so progress is kept per page: a run that stopped
    before it listed everything and committed nothing would start over at the same place."""
    fake = _three_projects()
    state: dict = {}

    rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS + 1, **SELECTION)

    assert [row["id"] for row in rows] == ["project:P1"]
    assert (stats["failed_budget"], stats["requests"]) == (1, LISTING_REQUESTS + 1)
    assert sorted(state["listing"]["done"]) == ["P1"] and state["known_ids"] == ["project:P1"]
    assert _tasks(rows) == {} and "pending" not in state, "rendering waits for a complete pass"


def test_every_request_counts_against_the_budget_and_the_listing_finishes_over_runs():
    fake = _three_projects(tasks_each=1)
    state: dict = {}
    seen: list[dict] = []

    for _ in range(20):
        fake.calls.clear()
        rows, stats = _sync(fake, state, max_requests=LISTING_REQUESTS + 1, **SELECTION)
        assert stats["requests"] <= LISTING_REQUESTS + 1
        seen += rows
        if stats["failed"] == 0:
            break

    assert stats["failed"] == 0 and "listing" not in state
    assert {"task:11", "task:21", "task:31"} <= set(_tasks(seen))


def test_a_stopped_run_forgets_nothing_and_a_later_complete_one_does():
    fake = _workspace(tasks=4)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("4")
    for gid in "123":
        fake.edit_task(gid)
    fake.fail("/tasks/2", 401)

    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == set() and stats["deleted"] == 0
    assert "task:4" in state["known_ids"], "a stop forgets nothing"
    fake.heal()
    rows, _ = _sync(fake, state)
    assert _tombstones(rows) == {"task:4"}


# --- check_active ---


def test_check_active_is_asked_before_every_request():
    fake = _workspace(tasks=2)
    asked: list[int] = []

    _sync(fake, {}, check_active=lambda: asked.append(len(fake.calls)))

    assert asked == list(range(len(fake.calls)))


def test_check_active_raising_stops_the_run_before_the_next_request():
    fake = _workspace()
    seen: list[int] = []

    def revoked():
        seen.append(1)
        if len(seen) == 4:
            raise PermissionError("access revoked")

    with pytest.raises(PermissionError):
        _sync(fake, {}, check_active=revoked)

    assert len(fake.calls) == 3


def test_an_oserror_from_check_active_is_not_retried_or_swallowed(monkeypatch):
    """The checkpoint runs outside the client's try/except, so a host bridge that fails with an
    OSError (a dropped socket) is neither retried like a network error nor read as a failed task."""
    fake = _workspace()
    sleeps: list[float] = []
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=sleeps.append)

    def bridge_down():
        raise OSError("host bridge down")

    with pytest.raises(OSError, match="host bridge down"):
        _sync(client, {}, check_active=bridge_down)

    assert fake.calls == [] and sleeps == []


def test_an_oserror_from_check_active_in_the_middle_of_a_task_is_not_a_failed_task(monkeypatch):
    fake = _workspace(tasks=2)
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=lambda _: None)
    seen: list[int] = []

    def flaky():
        seen.append(1)
        if len(seen) == LISTING_REQUESTS + 2:
            raise OSError("host bridge down")

    state: dict = {}
    with pytest.raises(OSError):
        _sync(client, state, check_active=flaky)

    assert state == {}


def test_a_request_the_budget_refuses_is_not_sent_and_asks_the_host_nothing():
    fake = _workspace()
    asked: list[int] = []

    _, stats = _sync(fake, {}, max_requests=3, check_active=lambda: asked.append(1))

    assert stats["failed_budget"] == 1
    assert len(asked) == len(fake.calls) == 3


def test_a_retried_request_asks_the_host_per_attempt_and_counts_once_against_the_budget(
    monkeypatch,
):
    fake = _workspace(tasks=2)
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    _, plain = _sync(AsanaClient(TOKEN, sleep=lambda _: None), {})
    fake.calls.clear()
    fake.fail("/tasks/2", 500, times=1)
    asked: list[int] = []

    _, stats = _sync(
        AsanaClient(TOKEN, sleep=lambda _: None),
        {},
        max_requests=plain["requests"],
        check_active=lambda: asked.append(1),
    )

    assert stats["requests"] == plain["requests"] and "failed_budget" not in stats
    assert len(fake.calls) == len(asked) == plain["requests"] + 1


def test_a_host_that_revokes_during_a_tasks_retry_stops_the_run_and_loses_that_task(monkeypatch):
    """The read of task 2 answers 500 once; the host refuses the retry. The run stops with the
    host's own error, task 2 is not emitted and the state is as the run found it."""
    fake = _workspace(tasks=2)
    fake.fail("/tasks/2", 500, times=1)
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    sleeps: list[float] = []
    client = AsanaClient(TOKEN, sleep=sleeps.append)
    error = PermissionError("access revoked")

    def check():
        if fake.calls and fake.calls[-1][0] == "/tasks/2":
            raise error

    state: dict = {}
    emitted: list[str] = []
    with pytest.raises(PermissionError) as raised:
        for row in _iter_rows(client, state, new_stats(), project_gids=["P1"], check_active=check):
            emitted.append(row["id"])

    assert raised.value is error
    assert "task:1" in emitted and "task:2" not in emitted
    assert [path for path, _ in fake.calls].count("/tasks/2") == 1 and sleeps == []
    assert state == {}


# --- the counts a host reads ---


def _exits():
    """One run for each way a run can end: ``(name, fake, state, kwargs, raises)``."""
    stopped = _workspace(tasks=3)
    stopped.fail("/tasks/2", 401)
    limited = _workspace()
    limited.fail("/tasks", 429)
    broken = _workspace()
    broken.fail("/events", 500)
    shaped = _workspace()
    shaped.answers["/tasks"] = {"data": "x"}
    rejected = _workspace()
    rejected.fail("/projects", 401)
    unavailable = _workspace(tasks=4)
    for gid in "123":
        unavailable.fail(f"/tasks/{gid}", 500)
    return [
        ("completed", _workspace(), {}, False),
        ("401 after rows", stopped, {}, False),
        ("429", limited, {}, False),
        ("budget", _workspace(), {"max_requests": TASK_REQUESTS}, False),
        ("outage", unavailable, {}, False),
        ("listing 500", broken, {}, True),
        ("wrong shape", shaped, {}, True),
        ("401 first", rejected, {}, True),
    ]


@pytest.mark.parametrize("index", range(8))
def test_every_count_is_present_as_a_non_negative_int_on_every_way_a_run_ends(index):
    """A host sums ``source_counts`` into its sync result and keeps only non-negative ints."""
    _name, fake, kwargs, raises = _exits()[index]
    stats = new_stats()
    state: dict = {}

    if raises:
        with pytest.raises(Exception, match="Asana"):
            list(_iter_rows(fake, state, stats, project_gids=["P1"], **kwargs))
    else:
        list(_iter_rows(fake, state, stats, project_gids=["P1"], **kwargs))

    assert set(new_stats()) <= set(stats) <= set(new_stats()) | REASONS
    assert all(type(value) is int and value >= 0 for value in stats.values())
    assert len(_reasons(stats)) <= 1, "at most one reason"
    assert (stats["failed"] > 0) == (index > 0), "failed says whether anything went wrong"
    assert stats["kept"] == len(state.get("retry", [])) and stats["pending"] == len(
        state.get("pending", [])
    )
    if _reasons(stats):
        assert stats["failed"] >= 1


def test_failed_counts_failures_and_never_the_kept_or_pending_tasks():
    fake = _workspace(tasks=6)
    for gid in "23":
        fake.fail(f"/tasks/{gid}", 403)
    fake.fail("/tasks/5", 401)

    _, stats = _sync(fake, {})

    # tasks 2 and 3 failed, the 401 on task 5 is the one refused request; 4 is read, 5 and 6 wait
    assert (stats["failed"], stats["kept"], stats["pending"]) == (3, 2, 2)


# --- the factory ---


def test_the_resource_is_a_merge_resource_tagged_for_document_ingestion():
    resource = asana_source(service=_workspace(), resource_name=TABLE, project_gids=["P1"])

    assert resource.name == TABLE
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "asana"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == TABLE
    hints = resource.compute_table_schema()
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True
    assert resource.cognee_sync_stats == {}, "filled when the run starts"


@pytest.mark.parametrize(
    "selection",
    [[], (), "", "1201234567890123", None, {"1"}, [1], [""], [" 1"], ["1", ""], ["a/b"], ["../x"]],
)
def test_a_selection_that_is_not_a_non_empty_list_of_gid_strings_is_refused(selection):
    """An empty selection means nothing, never everything: syncing it would tombstone every task."""
    with pytest.raises(ValueError, match="project_gids must be a non-empty list or tuple"):
        asana_source(service=_workspace(), resource_name=TABLE, project_gids=selection)


def test_the_selection_is_sorted_and_deduplicated(monkeypatch):
    fake = _workspace(tasks=0)
    fake.add_project("P2")
    seen: list[list[str]] = []
    original = asana_module._iter_rows
    monkeypatch.setattr(
        asana_module,
        "_iter_rows",
        lambda *a, **k: seen.append(k["project_gids"]) or original(*a, **k),
    )

    resource = asana_source(service=fake, resource_name=TABLE, project_gids=("P2", "P1", "P2"))
    list(resource)

    assert seen == [["P1", "P2"]]


@pytest.mark.parametrize(
    "name", ["_dlt_loads", "_dlt_pipeline_state", "_dlt_version", "_DLT_Loads", "__dlt_x", "_dlt_a"]
)
def test_a_name_dlt_keeps_for_itself_is_refused(name):
    with pytest.raises(ValueError, match="_dlt_"):
        asana_source(service=_workspace(), resource_name=name, project_gids=["P1"])


@pytest.mark.parametrize("name", ["", "  ", None, 7])
def test_a_blank_resource_name_is_refused(name):
    with pytest.raises(ValueError, match="resource_name"):
        asana_source(service=_workspace(), resource_name=name, project_gids=["P1"])


@pytest.mark.parametrize("name", ["asana_dlt", "_dlt", "asana", "a" * 70])
def test_other_names_are_accepted(name):
    asana_source(service=_workspace(), resource_name=name, project_gids=["P1"])


def test_there_is_no_default_for_the_name_or_the_selection():
    with pytest.raises(TypeError, match="resource_name"):
        asana_source(service=_workspace(), project_gids=["P1"])
    with pytest.raises(TypeError, match="project_gids"):
        asana_source(service=_workspace(), resource_name=TABLE)


def test_a_token_or_a_service_is_needed_and_a_bad_token_fails_at_build_time():
    with pytest.raises(ValueError, match="needs a service or an access_token"):
        asana_source(resource_name=TABLE, project_gids=["P1"])
    with pytest.raises(ValueError, match="invalid format") as raised:
        asana_source(resource_name=TABLE, project_gids=["P1"], access_token="SEC RET")
    assert "SEC" not in str(raised.value)


def test_the_token_is_never_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ASANA_ACCESS_TOKEN", TOKEN)

    with pytest.raises(ValueError, match="needs a service or an access_token"):
        asana_source(resource_name=TABLE, project_gids=["P1"])


def test_building_the_source_makes_no_request(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the factory touched the network")

    monkeypatch.setattr("requests.request", refuse)

    asana_source(resource_name=TABLE, project_gids=["P1"], access_token=TOKEN)


def test_refuses_a_core_that_does_not_read_per_row_node_sets(monkeypatch):
    monkeypatch.setattr(asana_module.dlt_utils, "DOCUMENT_SYNC_VERSION", 1)

    with pytest.raises(RuntimeError, match=r"DOCUMENT_SYNC_VERSION >= 2"):
        asana_source(service=_workspace(), resource_name=TABLE, project_gids=["P1"])


def test_a_listed_gid_that_could_reach_another_endpoint_is_refused():
    fake = _workspace()
    fake.answers["/tasks"] = {
        "data": [{"gid": "1/../../projects/P1", "modified_at": "2026-10-01T10:00:00.000Z"}]
    }

    with pytest.raises(AsanaResponseError, match="usable gid"):
        _sync(fake, {})

    assert [path for path, _ in fake.calls if path.startswith("/tasks/")] == []


# --- through a real dlt pipeline ---


def _pipeline(tmp_path):
    import dlt

    return dlt.pipeline(
        pipeline_name="asana_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'a.db'}"),
        dataset_name="asana",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


def _source(service, name: str = TABLE, **kwargs):
    kwargs.setdefault("project_gids", ["P1"])
    return asana_source(service=service, resource_name=name, **kwargs)


def _run(tmp_path, source, *, nothing_staged_yet: bool = False) -> dict[str, str]:
    """Run ``source`` the way ingest_dlt_source does; return the staged ``id -> content``.

    A run cut before its first row creates no table: only a test that says
    ``nothing_staged_yet`` may meet that, and gets ``{}``.
    """
    from dlt.destinations.exceptions import DatabaseUndefinedRelation

    pipeline = _pipeline(tmp_path)
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id, content FROM "{source.name}" ORDER BY id'
    try:
        with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
            return {row[0]: row[1] for row in cursor.fetchall()}
    except DatabaseUndefinedRelation:
        if nothing_staged_yet:
            return {}
        raise


def test_edits_and_deletes_survive_between_real_dlt_runs(tmp_path):
    fake = _workspace()
    first = _source(fake)
    assert list(_run(tmp_path, first)) == ["project:P1", "task:1", "task:2", "task:3"]
    assert first.cognee_sync_stats["failed"] == 0

    quiet = _source(fake)
    _run(tmp_path, quiet)
    assert quiet.cognee_sync_stats["skipped"] == quiet.cognee_sync_stats["scanned"] == 3

    fake.edit_task("1", notes="Rewritten.")
    fake.delete_task("3")
    forget = _source(fake)
    staged = _run(tmp_path, forget)

    assert list(staged) == ["project:P1", "task:1", "task:2"]
    assert "Rewritten." in staged["task:1"]
    assert forget.cognee_sync_stats["deleted"] == 1


def test_the_node_set_reaches_the_destination_and_the_structure_stays_unset(tmp_path):
    """ingest_dlt_source adds the node set and structure columns as json hints to every document
    source. This source fills the node set (the project gids) and never the structure."""
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1", "P2"))
    fake.add_task("2", projects=("P2",))
    _run(tmp_path, _source(fake, project_gids=["P1", "P2"]))
    pipeline = _pipeline(tmp_path)

    with pipeline.sql_client() as sql:
        with sql.execute_query(f'PRAGMA table_info("{TABLE}")') as cursor:
            columns = {row[1] for row in cursor.fetchall() if not row[1].startswith("_dlt_")}
        query = f'SELECT id, cognee_node_set, cognee_structure FROM "{TABLE}" ORDER BY id'
        with sql.execute_query(query) as cursor:
            stored = cursor.fetchall()

    assert columns == ROW_COLUMNS | {"cognee_structure"}
    assert {row[0]: json.loads(row[1]) for row in stored} == {
        "project:P1": ["asana:P1"],
        "project:P2": ["asana:P2"],
        "task:1": ["asana:P1", "asana:P2"],
        "task:2": ["asana:P2"],
    }
    assert all(row[2] is None for row in stored)


def test_a_comment_edit_reaches_the_destination_while_modified_at_stays_put(tmp_path):
    fake = _workspace(tasks=1)
    fake.add_comment("1", "Grace", "first")
    _run(tmp_path, _source(fake))
    fake.edit_comment("1", 0, "edited")

    staged = _run(tmp_path, _source(fake))

    assert "- Grace: edited" in staged["task:1"]


def test_unreadable_tasks_never_block_new_ones_across_real_runs(tmp_path):
    fake = _workspace(tasks=6)
    for gid, status in (("2", 403), ("3", 404), ("4", 403)):
        fake.fail(f"/tasks/{gid}", status)

    for run in range(4):
        fake.add_task(f"9{run}")
        source = _source(fake)
        staged = _run(tmp_path, source)
        assert f"task:9{run}" in staged, "a new task is ingested whatever the others do"
        assert {"task:2", "task:3", "task:4"}.isdisjoint(staged)
        assert (source.cognee_sync_stats["failed"], source.cognee_sync_stats["kept"]) == (3, 3)
        assert _reasons(source.cognee_sync_stats) == set()

    fake.heal()
    staged = _run(tmp_path, _source(fake))
    assert {"task:2", "task:3", "task:4"} <= set(staged)


def test_a_malformed_task_keeps_its_document_and_never_blocks_new_ones_across_real_runs(tmp_path):
    fake = _workspace()
    _run(tmp_path, _source(fake))
    fake.edit_task("2", notes="Rewritten.")
    fake.answers["/tasks/2/stories"] = {"data": [1]}

    for run in range(3):
        fake.add_task(f"9{run}")
        source = _source(fake)
        staged = _run(tmp_path, source)
        assert f"task:9{run}" in staged, "a new task is ingested whatever the others do"
        assert "Body 2." in staged["task:2"], "the old document stays"
        assert (source.cognee_sync_stats["failed"], source.cognee_sync_stats["kept"]) == (1, 1)
        assert _reasons(source.cognee_sync_stats) == set()

    del fake.answers["/tasks/2/stories"]
    staged = _run(tmp_path, _source(fake))
    assert "Rewritten." in staged["task:2"]


def test_an_ingested_task_that_cannot_be_read_keeps_its_document_until_it_leaves(tmp_path):
    fake = _workspace()
    _run(tmp_path, _source(fake))
    fake.edit_task("2", notes="Rewritten.")
    stamp = fake.tasks["2"]["modified_at"]
    fake.fail("/tasks/2", 403)

    for _ in range(5):
        source = _source(fake)
        staged = _run(tmp_path, source)
        assert "Body 2." in staged["task:2"], "the old document stays"
        assert source.cognee_sync_stats["deleted"] == 0

    fake.heal()
    assert fake.tasks["2"]["modified_at"] == stamp
    staged = _run(tmp_path, _source(fake))
    assert "Rewritten." in staged["task:2"], "read again with modified_at unchanged"

    fake.edit_task("2")
    fake.fail("/tasks/2", 404)
    _run(tmp_path, _source(fake))
    fake.delete_task("2")
    assert "task:2" not in _run(tmp_path, _source(fake)), "a task that left the listing goes"


def test_a_slow_task_with_a_retry_after_never_stops_the_others_across_real_runs(
    tmp_path, monkeypatch
):
    fake = _workspace(tasks=3)
    fake.fail("/tasks/2", 500, headers={"Retry-After": "86400"})
    sleeps: list[float] = []
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=sleeps.append)

    for run in range(1, 4):
        fake.add_task(f"3{run}")
        staged = _run(tmp_path, _source(client))
        assert f"task:3{run}" in staged and "task:1" in staged and "task:3" in staged

    assert sleeps == [1, 2] * 3


def test_two_sources_with_their_own_resource_names_stay_apart_in_one_dataset(tmp_path):
    team_a, team_b = _workspace(), FakeAsana()
    team_b.add_project("P1", "Other")
    team_b.add_task("1", "Elsewhere")
    team_b.add_task("2", "Elsewhere too")

    def run(fake, name):
        return list(_run(tmp_path, _source(fake, name)))

    assert run(team_a, "asana_a") == ["project:P1", "task:1", "task:2", "task:3"]
    assert run(team_b, "asana_b") == ["project:P1", "task:1", "task:2"]

    team_b.delete_task("2")
    assert run(team_b, "asana_b") == ["project:P1", "task:1"]
    assert run(team_a, "asana_a") == ["project:P1", "task:1", "task:2", "task:3"]


def test_a_clean_stop_resumes_from_pending_and_loses_nothing_that_changed_meanwhile(tmp_path):
    fake = _workspace(tasks=6)
    _run(tmp_path, _source(fake))
    for gid in "23456":
        fake.edit_task(gid, notes=f"Edit one of {gid}.")
    fake.fail("/tasks/4", 401)
    cut = _source(fake)

    staged = _run(tmp_path, cut)

    stats = cut.cognee_sync_stats
    assert (stats["failed_auth"], stats["deleted"], stats["pending"]) == (1, 0, 3)
    assert "Edit one of 3." in staged["task:3"] and "Body 4." in staged["task:4"]
    assert set(staged) == {"project:P1", *(f"task:{gid}" for gid in "123456")}, "nothing forgotten"

    # while 4, 5 and 6 wait: one is edited again, one is deleted, one moves out of the selection
    fake.heal()
    fake.edit_task("4", notes="Edit two of 4.")
    fake.delete_task("5")
    fake.add_project("P9", "Elsewhere")
    fake.move_task("6", ["P9"])
    done = _source(fake)

    staged = _run(tmp_path, done)

    assert done.cognee_sync_stats["failed"] == 0 and done.cognee_sync_stats["pending"] == 0
    assert "Edit two of 4." in staged["task:4"], "the latest edit, not the one that was pending"
    assert set(staged) == {"project:P1", "task:1", "task:2", "task:3", "task:4"}
    assert done.cognee_sync_stats["deleted"] == 2


def test_a_rejected_token_with_nothing_emitted_raises_through_dlt_and_the_next_run_is_whole(
    tmp_path,
):
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = _workspace()
    fake.fail("/projects", 401)
    source = _source(fake)

    with pytest.raises(PipelineStepFailed) as raised:
        _run(tmp_path, source)

    chain = []
    error = raised.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(error, AsanaAuthError) for error in chain)
    assert (source.cognee_sync_stats["failed"], source.cognee_sync_stats["failed_auth"]) == (1, 1)
    fake.heal()
    assert list(_run(tmp_path, _source(fake))) == ["project:P1", "task:1", "task:2", "task:3"]


def test_check_active_can_stop_a_real_run_and_nothing_is_loaded(tmp_path):
    from dlt.pipeline.exceptions import PipelineStepFailed

    def revoked():
        raise PermissionError("access revoked")

    fake = _workspace()
    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, _source(fake, check_active=revoked))

    assert fake.calls == []


def test_a_selection_change_reaches_the_destination(tmp_path):
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1", "P2"))
    fake.add_task("2", projects=("P2",))
    both = _run(tmp_path, _source(fake, project_gids=["P1", "P2"]))
    assert "Project: Blog" in both["task:1"] and "task:2" in both

    only_p1 = _run(tmp_path, _source(fake, project_gids=["P1"]))

    assert "Blog" not in only_p1["task:1"]
    assert list(only_p1) == ["project:P1", "task:1"]


def test_a_dlt_reserved_name_is_stopped_by_the_factory_not_by_the_load(tmp_path):
    """dlt owns ``_dlt_loads``: a resource of that name fails the load after the walk."""
    for name in ("_dlt_loads", "_dlt_pipeline_state", "_dlt_version"):
        with pytest.raises(ValueError, match="_dlt_"):
            _source(_workspace(), name)


def test_a_network_failure_on_a_task_is_kept_like_any_other_failure(monkeypatch):
    fake = _workspace(tasks=2)
    fake.fail("/tasks/1", None, message=TOKEN, times=3)
    sleeps: list[float] = []
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=sleeps.append)

    rows, stats = _sync(client, {})

    assert set(_tasks(rows)) == {"task:2"} and stats["failed"] == 1
    assert sleeps == [1, 2]
    assert TOKEN not in repr(rows)


@pytest.mark.parametrize("status", [403, 404, 500, 401])
def test_a_run_through_the_real_client_leaks_the_token_nowhere(monkeypatch, caplog, status):
    """The service= fakes never touch the code that builds a message, so this goes through the real
    client with only the transport faked. Asana echoes the token in every error explanation."""
    import logging

    fake = _workspace(tasks=3)
    fake.fail("/tasks/2", status, message=f"Bad header: Bearer {TOKEN}")
    monkeypatch.setattr("requests.request", serve_over_http(fake))
    client = AsanaClient(TOKEN, sleep=lambda _: None)
    state: dict = {}
    stats = new_stats()

    with caplog.at_level(logging.DEBUG):
        rows = list(_iter_rows(client, state, stats, project_gids=["P1"]))

    everything = " ".join([repr(rows), repr(state), repr(stats), caplog.text])
    assert TOKEN not in everything


def test_a_task_queued_for_a_read_survives_its_project_being_unresolved_for_a_run(tmp_path):
    """Task 2 is edited early and task 1 much later, so the cursor moves past task 2: only its
    place in ``pending`` can bring its edit in once the project answers again."""
    fake = _workspace(tasks=2)
    _run(tmp_path, _source(fake))
    fake.edit_task("2", notes="Rewritten early.")
    fake._clock += 1000
    fake.edit_task("1", notes="Rewritten late.")
    cut = _source(fake, max_requests=LISTING_REQUESTS + TASK_REQUESTS)
    staged = _run(tmp_path, cut)
    assert "Rewritten late." in staged["task:1"] and "Rewritten early." not in staged["task:2"]

    fake.fail("/projects/P1", 403)
    out = _source(fake)
    _run(tmp_path, out)
    assert out.cognee_sync_stats["unresolved"] == 1
    fake.heal()

    back = _source(fake)
    staged = _run(tmp_path, back)

    assert "Rewritten early." in staged["task:2"]


def test_a_small_rate_limit_still_lists_a_selection_over_several_real_runs(tmp_path):
    """Each project costs three listing requests and the window serves seven: a run lists two
    projects and is cut inside the third, and the next run continues there. Before, every run
    began at the first project and none ever got past the second."""
    fake = _three_projects(tasks_each=1)
    fake.rate_limit(7)
    firsts: list[str] = []
    staged: dict[str, str] = {}

    for _ in range(20):
        fake.new_window()
        fake.calls.clear()
        source = _source(fake, project_gids=["P1", "P2", "P3"])
        staged = _run(tmp_path, source)
        firsts.append(_projects_requested(fake)[0])
        if source.cognee_sync_stats["failed"] == 0:
            break

    assert source.cognee_sync_stats["failed"] == 0, "the selection is eventually synced"
    assert set(staged) == {
        "project:P1",
        "project:P2",
        "project:P3",
        "task:11",
        "task:21",
        "task:31",
    }
    assert firsts[:2] == ["P1", "P3"], "the second run resumes at the project the first was cut in"


def test_a_cut_listing_forgets_nothing_through_a_real_dlt_run(tmp_path):
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1",))
    fake.add_task("2", projects=("P2",))
    both = {"project_gids": ["P1", "P2"]}
    _run(tmp_path, _source(fake, **both))
    fake.delete_task("2")
    fake.fail("/tasks", 429, after=1)

    cut = _source(fake, **both)
    staged = _run(tmp_path, cut)

    assert set(staged) == {"project:P1", "project:P2", "task:1", "task:2"}
    assert (cut.cognee_sync_stats["failed_rate_limit"], cut.cognee_sync_stats["deleted"]) == (1, 0)
    fake.heal()

    done = _source(fake, **both)
    staged = _run(tmp_path, done)

    assert set(staged) == {"project:P1", "project:P2", "task:1"}
    assert done.cognee_sync_stats["deleted"] == 1


def test_a_dlt_resource_whose_selection_changes_mid_pass_starts_the_pass_over(tmp_path):
    fake = _three_projects()
    fake.fail("/tasks", 429, after=2)
    _run(tmp_path, _source(fake, **SELECTION))
    fake.heal()
    fake.calls.clear()

    source = _source(fake, project_gids=["P1", "P2"])
    staged = _run(tmp_path, source)

    assert [p.split("/")[2] for p, _ in fake.calls if re.fullmatch(r"/projects/\w+", p)] == [
        "P1",
        "P2",
    ]
    assert source.cognee_sync_stats["failed"] == 0
    assert set(staged) == {"project:P1", "project:P2"} | {
        f"task:{n}{m}" for n in (1, 2) for m in (1, 2)
    }


def test_a_big_listing_makes_progress_in_every_run_and_finishes_over_several(tmp_path):
    """600 projects with one task each cost 1800 listing requests, more than one run's 1500. The
    listing is budgeted and keeps its progress, so the first run only lists 500 projects, the
    second lists the rest and renders what is left of its budget, and a later pass renders more
    until everything is staged."""
    fake = FakeAsana()
    gids = [f"P{n}" for n in range(600)]
    for n, gid in enumerate(gids):
        fake.add_project(gid, f"Project {n}", notes="")
        fake.add_task(f"{n + 1}", f"Task {n}", projects=(gid,))

    first = _source(fake, project_gids=gids)
    staged = _run(tmp_path, first)
    assert len([key for key in staged if key.startswith("project:")]) == 500
    assert not [key for key in staged if key.startswith("task:")], "rendering waits"
    assert first.cognee_sync_stats["requests"] == 1500

    for _ in range(5):
        source = _source(fake, project_gids=gids)
        staged = _run(tmp_path, source)
        if len({key for key in staged if key.startswith("task:")}) == 600:
            break

    assert len({key for key in staged if key.startswith("task:")}) == 600
    assert source.cognee_sync_stats["pending"] == 0


@pytest.mark.parametrize("bad", [0, -5, True, False, 1.5, "10", None])
def test_max_requests_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="max_requests must be a positive int"):
        asana_source(
            service=_workspace(), resource_name=TABLE, project_gids=["P1"], max_requests=bad
        )


@pytest.mark.parametrize("bad", [1, 2])
def test_a_budget_that_cannot_read_one_task_is_refused(bad):
    """A task costs three requests, so a budget of 1 or 2 would run for ever and read nothing."""
    with pytest.raises(ValueError, match="at least 3"):
        asana_source(
            service=_workspace(), resource_name=TABLE, project_gids=["P1"], max_requests=bad
        )


def test_a_budget_of_exactly_one_task_is_accepted():
    asana_source(
        service=_workspace(), resource_name=TABLE, project_gids=["P1"], max_requests=TASK_REQUESTS
    )


def test_a_task_gid_is_never_logged_and_kept_tasks_are_one_warning_a_run(caplog):
    import logging

    fake = _workspace(tasks=3)
    fake.add_task("7731234", "Hidden")
    fake.fail("/tasks/7731234", 403)
    fake.fail("/tasks/2", 404)

    with caplog.at_level(logging.DEBUG):
        _sync(fake, {})

    assert "7731234" not in caplog.text
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "2 task(s) could not be read" in warnings[0].getMessage()


def test_a_run_that_raises_reports_its_own_kept_and_pending_not_the_last_runs():
    fake = _workspace(tasks=3)
    fake.fail("/tasks/2", 403)
    state: dict = {}
    _, stats = _sync(fake, state)
    assert (stats["kept"], state["retry"]) == (1, ["2"])
    fake.fail("/projects/P1", 500)
    stats = new_stats()

    with pytest.raises(AsanaAPIError):
        list(_iter_rows(fake, state, stats, project_gids=["P1"]))

    assert (stats["kept"], stats["pending"]) == (0, 0)
    assert state["retry"] == ["2"], "the state itself is untouched"


def test_a_token_is_stripped_like_linear_and_apollo_do_and_sent_clean(monkeypatch):
    sent = []

    def request(method, url, **kwargs):
        sent.append(kwargs["headers"]["Authorization"])
        return Resp(200, {"data": []})

    monkeypatch.setattr("requests.request", request)

    AsanaClient(f"  {TOKEN}\n").request("GET", "/tasks")

    assert sent == [f"Bearer {TOKEN}"]


def test_a_task_moved_between_projects_is_seen_through_events_with_modified_at_unchanged(tmp_path):
    """Asana's modified_at does not follow a project membership change: the added and removed
    events are the only signal, and the task must not be forgotten or left with the old line."""
    fake = FakeAsana()
    fake.add_project("P1", "Alpha")
    fake.add_project("P2", "Beta")
    # the mover is far older than P2's newest task, so only the events can bring it back
    fake.add_task("1", "Mover", projects=("P1",), modified_at="2026-09-01T00:00:00.000Z")
    fake.add_task("2", "Stays", projects=("P1",))
    fake.add_task("3", "Beta resident", projects=("P2",))
    both = {"project_gids": ["P1", "P2"]}
    first = _run(tmp_path, _source(fake, **both))
    assert "Project: Alpha" in first["task:1"]
    stamp = fake.tasks["1"]["modified_at"]
    fake.move_task("1", ["P2"])

    source = _source(fake, **both)
    staged = _run(tmp_path, source)

    assert fake.tasks["1"]["modified_at"] == stamp
    assert "Project: Beta" in staged["task:1"] and "Alpha" not in staged["task:1"]
    assert set(staged) == {"project:P1", "project:P2", "task:1", "task:2", "task:3"}
    assert source.cognee_sync_stats["deleted"] == 0
