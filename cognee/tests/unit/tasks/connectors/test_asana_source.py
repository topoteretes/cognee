# Adapted from topoteretes/cognee-community (Apache-2.0), PR #349, commit ef32699 (AbdulWasih05).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""An Asana task is re-extracted only when its rendered text changes, and nothing but a complete
listing ever forgets one.

The walk is driven with a plain dict as dlt resource state, and then through a real ``dlt``
pipeline into SQLite. What a task that cannot be read does is the heart of it: while it is still
listed it keeps its old document, is asked again on the next run after every fresh task, never
counts toward an outage and never stops another task. A run that stops (a rejected token after
rows, a rate limit, a spent budget, an outage) keeps what it emitted, queues the rest and
forgets nothing; a 429 is never waited out.
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
    _documents_for,
    _event_task_gids,
    _iter_rows,
    _renamed,
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
    """The gids of the tasks read, in the order their task answers were asked for."""
    return [p.split("/")[2] for p, _ in fake.calls if re.fullmatch(r"/tasks/\w+", p)]


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
    base = {"name": "T", "notes": "body", "modified_at": "2026-10-01T00:00:00.000Z", "num_likes": 0}
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
    row = render_task("1", {"completed": True}, [], [], [])

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
    row = render_task("1", {"completed": True, "completed_at": completed_at}, [], [], [])

    assert row["content"] == f"Completed: yes\n{line}"


def test_an_open_task_renders_no_completion_day_even_if_asana_still_sends_one():
    """Reopening a task changes its text, which is right: the line goes with ``Completed: yes``."""
    done = {"completed": True, "completed_at": "2026-10-09T19:14:27.123Z"}
    reopened = {"completed": False, "completed_at": None}
    stale = {"completed": False, "completed_at": "2026-10-09T19:14:27.123Z"}

    assert "Completed on" in render_task("1", done, [], [], [])["content"]
    assert render_task("1", reopened, [], [], [])["content"] == "Completed: no"
    assert render_task("1", stale, [], [], [])["content"] == "Completed: no"


@pytest.mark.parametrize(
    "completed_at",
    [None, "", "yesterday", "2026-13-45T00:00:00.000Z", 20261009, ["2026-10-09"], "2026-10-09"],
)
def test_a_completed_at_that_cannot_be_read_renders_no_line_and_never_raises(completed_at):
    row = render_task("1", {"completed": True, "completed_at": completed_at}, [], [], [])

    assert row["content"] == "Completed: yes"


def test_a_completed_task_without_completed_at_renders_no_line():
    assert render_task("1", {"completed": True}, [], [], [])["content"] == "Completed: yes"


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


def test_an_events_feed_whose_token_does_not_advance_raises_instead_of_looping():
    fake = _workspace()
    fake.answers["/events"] = {"data": [], "sync": "same", "has_more": True}
    state = {"projects": {"P1": {"cursor": fake.stamp(), "sync": "same", "seen": {}}}}

    with pytest.raises(AsanaResponseError, match="did not advance"):
        _sync(fake, state)


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
    ("prefix", "status"), [("/tasks", 500), ("/events", 403), ("/projects/P1", 404)]
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
    fake.stuck_offset = "offset:2"

    with pytest.raises(AsanaResponseError, match="offset came back twice"):
        _sync(fake, {})


@pytest.mark.parametrize("answer", [{}, {"data": []}, {"data": "x"}])
def test_a_task_answer_of_the_wrong_shape_aborts_the_run(answer):
    fake = _workspace()
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2")
    before = copy.deepcopy(state)
    fake.answers["/tasks/2"] = answer

    with pytest.raises(AsanaResponseError):
        _sync(fake, state)

    assert state == before


def test_a_comment_listing_of_the_wrong_shape_aborts_the_run():
    fake = _workspace()
    fake.answers["/tasks/1/stories"] = {"data": [1, 2]}

    with pytest.raises(AsanaResponseError, match="not an object"):
        _sync(fake, {})


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


def test_retries_go_last_and_never_count_toward_an_outage():
    fake = _workspace(tasks=5)
    for gid in "2345":
        fake.fail(f"/tasks/{gid}", 403)
    state: dict = {}
    _sync(fake, state)
    fake.heal()
    for gid in "2345":
        fake.fail(f"/tasks/{gid}", 500)
    fake.add_task("9")
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert _fetched(fake)[0] == "9", "a fresh task is rendered before every retry"
    assert set(_tasks(rows)) == {"task:9"}
    assert (stats["failed"], _reasons(stats)) == (4, set()), "four retries are no outage"
    assert state["retry"] == ["2", "3", "4", "5"] and state["pending"] == []


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

    assert rows == [] and state == before
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
        assert rows == [] and state == {}


def test_a_spent_budget_stops_cleanly_and_the_next_run_finishes():
    fake = _workspace(tasks=4)
    state: dict = {}
    budget = TASK_REQUESTS

    rows, stats = _sync(fake, state, max_requests=budget)

    assert set(_tasks(rows)) == {"task:1"}
    # the budget counts the requests that read tasks; the listing is on top of it
    assert (stats["failed"], stats["failed_budget"]) == (1, 1)
    assert stats["requests"] == LISTING_REQUESTS + budget
    assert state["pending"] == ["2", "3", "4"]
    rows, stats = _sync(fake, state)
    assert set(_tasks(rows)) == {"task:2", "task:3", "task:4"} and stats["failed"] == 0


def test_a_budget_spent_in_the_middle_of_a_task_leaves_that_task_pending():
    fake = _workspace(tasks=2)
    state: dict = {}

    rows, stats = _sync(fake, state, max_requests=2)

    assert _tasks(rows) == {} and stats["failed_budget"] == 1
    assert state["pending"] == ["1", "2"] and state["known_ids"] == ["project:P1"]


def test_the_budget_never_stops_a_run_before_it_has_listed():
    """The listing is not budgeted: a run that stopped before it listed would commit nothing and
    start over at the same place for ever."""
    fake = _workspace()
    state: dict = {}

    rows, stats = _sync(fake, state, max_requests=1)

    assert [row["id"] for row in rows] == ["project:P1"]
    assert state["v"] == 1 and state["pending"] == ["1", "2", "3"]
    assert (stats["failed_budget"], stats["pending"]) == (1, 3)
    assert stats["requests"] == LISTING_REQUESTS + 1


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
    """The checkpoint runs outside the client's retry loop, so a host bridge that fails with an
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


def _run(tmp_path, source) -> dict[str, str]:
    """Run ``source`` the way ingest_dlt_source does; return the staged ``id -> content``."""
    pipeline = _pipeline(tmp_path)
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id, content FROM "{source.name}" ORDER BY id'
    with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
        return {row[0]: row[1] for row in cursor.fetchall()}


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


def test_a_big_listing_makes_progress_in_every_run_and_finishes_over_several(tmp_path):
    """600 projects with one task each cost 1800 listing requests a run: more than the budget. The
    listing is not budgeted, so each run still lists in full, renders what the budget allows and
    commits, and the next run continues. With the listing inside the budget every run stopped
    at the same place, committed nothing and rendered nothing, for ever."""
    fake = FakeAsana()
    gids = [f"P{n}" for n in range(600)]
    for n, gid in enumerate(gids):
        fake.add_project(gid, f"Project {n}", notes="")
        fake.add_task(f"{n + 1}", f"Task {n}", projects=(gid,))

    first = _source(fake, project_gids=gids)
    staged = _run(tmp_path, first)
    tasks = {key for key in staged if key.startswith("task:")}
    assert len(tasks) == 500, "1500 task requests, three a task"
    assert (first.cognee_sync_stats["failed_budget"], first.cognee_sync_stats["pending"]) == (
        1,
        100,
    )
    assert first.cognee_sync_stats["requests"] == 1800 + 1500

    second = _source(fake, project_gids=gids)
    staged = _run(tmp_path, second)

    assert len({key for key in staged if key.startswith("task:")}) == 600
    assert second.cognee_sync_stats["pending"] == 0 and second.cognee_sync_stats["failed"] == 0


@pytest.mark.parametrize("bad", [0, -5, True, False, 1.5, "10", None])
def test_max_requests_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="max_requests must be a positive int"):
        asana_source(
            service=_workspace(), resource_name=TABLE, project_gids=["P1"], max_requests=bad
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
