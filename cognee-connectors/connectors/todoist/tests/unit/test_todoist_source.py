"""The Todoist source: rendering, the sync_token cursor and forget-on-delete.

The walk is driven with a plain dict as state and ``FakeTodoist`` as the service, so the
whole ingest path — full sync, incremental sync, comment folding, deletions, selection,
the request budget — is tested without a network or a dlt pipeline; a separate section
runs the same source through a real dlt pipeline into SQLite.
"""

import copy

import pytest
from fake_todoist import FakeTodoist

from cognee_connector_todoist.client import TodoistAuthError, TodoistResponseError
from cognee_connector_todoist.source import (
    DEFAULT_MAX_REQUESTS,
    STATE_VERSION,
    _run_rows,
    new_stats,
    render_project,
    render_task,
    todoist_source,
)
from cognee_connectors_core.dlt_conventions import DOCUMENT_COLUMN_HINTS, NODE_SET_COLUMN

ROW_COLUMNS = {"id", "title", "content", "url", "_deleted", NODE_SET_COLUMN}
TOMBSTONE_COLUMNS = {"id", "_deleted"}


def _account(tasks=2, projects=("P1",)) -> FakeTodoist:
    fake = FakeTodoist()
    for pid in projects:
        fake.add_project(pid, name=f"Project {pid}")
    for number in range(1, tasks + 1):
        pid = projects[(number - 1) % len(projects)]
        fake.add_task(str(number), pid, content=f"Task {number}", description=f"Body {number}.")
    return fake


def _sync(fake, state: dict, *, project_ids=None, max_requests=1000):
    stats = new_stats()
    rows = list(
        _run_rows(
            fake,
            state,
            stats,
            project_ids=project_ids,
            max_requests=max_requests,
        )
    )
    return rows, stats


def _rows_of(rows):
    return {row["id"]: row for row in rows if not row["_deleted"]}


def _tombstones(rows):
    return sorted(row["id"] for row in rows if row["_deleted"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_a_task_renders_its_fields_and_comments():
    item = {
        "id": "1",
        "project_id": "P1",
        "content": "Write the launch post",
        "description": "Draft, then review with marketing.",
        "priority": 3,
        "labels": ["launch", "website"],
        "due": {"date": "2026-10-20", "string": "Oct 20"},
        "checked": 1,
        "completed_at": "2026-10-14T10:00:00Z",
        "updated_at": "2026-10-14T09:00:00Z",
    }
    comments = [
        {"id": "c2", "item_id": "1", "content": "Second", "posted_at": "2026-10-02T09:00:02Z"},
        {"id": "c1", "item_id": "1", "content": "First", "posted_at": "2026-10-02T09:00:01Z"},
    ]

    row = render_task("1", item, comments)

    assert row["id"] == "task:1"
    assert row["title"] == "Write the launch post"
    assert row["url"] == "https://app.todoist.com/app/task/1"
    assert row["_deleted"] is False
    assert row[NODE_SET_COLUMN] == ["todoist:P1"]
    assert row["content"] == (
        "Completed: yes\n"
        "Completed on: 2026-10-14\n"
        "Priority: high\n"
        "Due: 2026-10-20\n"
        "Labels: launch, website\n"
        "\n"
        "Draft, then review with marketing.\n"
        "\n"
        "Comments:\n"
        "- First\n"
        "- Second"
    )
    assert "updated_at" not in row["content"], "nothing volatile reaches the text"
    assert "2026-10-1T" not in row["content"].replace("2026-10-14", "")


def test_an_open_task_with_defaults_renders_a_minimal_document():
    row = render_task("1", {"id": "1", "project_id": "P1", "content": "", "priority": 1}, [])

    assert row["title"] == "Untitled task"
    assert row["content"] == "Completed: no\nPriority: normal"
    assert row[NODE_SET_COLUMN] == ["todoist:P1"]


def test_a_task_without_a_project_has_an_empty_node_set_that_core_does_not_hash():
    row = render_task("1", {"id": "1", "content": "x", "priority": 1}, [])

    assert row[NODE_SET_COLUMN] == []


def test_shuffled_comments_render_the_same_text():
    item = {"id": "1", "project_id": "P1", "content": "x", "priority": 1}
    comments = [
        {"id": "c1", "item_id": "1", "content": "A", "posted_at": "2026-10-02T09:00:01Z"},
        {"id": "c2", "item_id": "1", "content": "B", "posted_at": "2026-10-02T09:00:02Z"},
        {"id": "c3", "item_id": "1", "content": "C", "posted_at": "2026-10-02T09:00:00Z"},
    ]

    forward = render_task("1", item, comments)["content"]
    backward = render_task("1", item, comments[::-1])["content"]

    assert forward == backward


def test_a_project_row_names_only_itself():
    row = render_project("P1", {"id": "P1", "name": "Website launch"})

    assert row == {
        "id": "project:P1",
        "title": "Website launch",
        "content": "",
        "url": "",
        "_deleted": False,
        NODE_SET_COLUMN: ["todoist:P1"],
    }


# ---------------------------------------------------------------------------
# The sync cursor (through a plain dict as state)
# ---------------------------------------------------------------------------
def test_a_first_sync_emits_every_project_and_task_with_its_comments():
    fake = _account(tasks=2)
    fake.add_comment("1", "Looks good")
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert [row["id"] for row in rows] == ["project:P1", "task:1", "task:2"]
    assert all(set(row) == ROW_COLUMNS for row in rows), "exactly the document columns"
    task = _rows_of(rows)["task:1"]
    assert "Body 1." in task["content"]
    assert "- Looks good" in task["content"]
    assert state["v"] == STATE_VERSION == 1
    assert state["tasks"] == {"1": "P1", "2": "P1"}
    assert state["projects"] == ["P1"]
    assert state["selection"] is None
    assert state["sync_token"] == str(len(fake.journal))
    assert stats == {**new_stats(), "changes": 4, "rendered": 2, "requests": 3}


def test_a_quiet_second_sync_reads_one_request_and_renders_nothing():
    fake = _account()
    state: dict = {}
    _sync(fake, state)
    token = state["sync_token"]

    rows, stats = _sync(fake, state)

    assert rows == []
    assert state["sync_token"] == token, "the cursor moves only when there was something to see"
    assert stats["requests"] == 1
    assert stats["rendered"] == 0


def test_an_edited_task_is_rendered_again_and_only_that_one():
    fake = _account(tasks=3)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("2", content="Task 2, rewritten", description="New body.")

    rows, stats = _sync(fake, state)

    assert [row["id"] for row in rows] == ["task:2"]
    assert rows[0]["title"] == "Task 2, rewritten"
    assert "New body." in rows[0]["content"]
    assert stats["rendered"] == 1
    assert stats["requests"] == 2, "the sync call and the task's comments"


def test_a_new_task_is_rendered_incrementally_with_the_comments_it_grew():
    fake = _account(tasks=1)
    state: dict = {}
    _sync(fake, state)
    fake.add_task("9", "P1", content="Fresh", description="Just added.")
    fake.add_comment("9", "First in")

    rows, _ = _sync(fake, state)

    assert [row["id"] for row in rows] == ["task:9"]
    assert "- First in" in rows[0]["content"]


def test_a_comment_on_an_unchanged_task_renders_the_task_with_the_comment():
    fake = _account(tasks=1)
    state: dict = {}
    _sync(fake, state)
    fake.add_comment("1", "A late remark")

    rows, _ = _sync(fake, state)

    assert [row["id"] for row in rows] == ["task:1"]
    assert "- A late remark" in rows[0]["content"]
    assert ("get_task", "1") in fake.calls, "the feed carried the note, not the task's body"


def test_an_edited_comment_renders_its_task_with_the_new_text():
    fake = _account(tasks=1)
    fake.add_comment("1", "Before")
    state: dict = {}
    _sync(fake, state)
    fake.edit_comment("1", "c1", "After")

    rows, _ = _sync(fake, state)

    assert [row["id"] for row in rows] == ["task:1"]
    assert "- After" in rows[0]["content"]
    assert "Before" not in rows[0]["content"]


def test_a_deleted_task_is_tombstoned_and_leaves_the_known_set():
    fake = _account(tasks=2)
    state: dict = {}
    _sync(fake, state)
    fake.delete_task("1")

    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == ["task:1"]
    tombstone = rows[0]
    assert set(tombstone) == TOMBSTONE_COLUMNS
    assert state["tasks"] == {"2": "P1"}
    assert stats["deleted"] == 1


def test_a_deleted_project_tombstones_it_and_every_task_of_it():
    fake = _account(tasks=2)
    fake.add_project("P2", name="Other")
    fake.add_task("3", "P2", content="Task 3")
    state: dict = {}
    _sync(fake, state)
    fake.delete_project("P1")

    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == ["project:P1", "task:1", "task:2"]
    assert state["tasks"] == {"3": "P2"}
    assert state["projects"] == ["P2"]


def test_a_moved_task_is_rendered_again_under_its_new_project():
    fake = _account(tasks=1)
    fake.add_project("P2")
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", project_id="P2")

    rows, _ = _sync(fake, state)

    assert [row["id"] for row in rows] == ["task:1"]
    assert rows[0][NODE_SET_COLUMN] == ["todoist:P2"]


def test_a_completed_task_names_the_day_and_reopening_removes_the_line():
    fake = _account(tasks=1)
    state: dict = {}
    _sync(fake, state)
    fake.complete_task("1")

    rows, _ = _sync(fake, state)
    assert "Completed: yes\nCompleted on: 2026-10-01" in rows[0]["content"]

    fake.edit_task("1", checked=0, completed_at=None)
    rows, _ = _sync(fake, state)
    assert rows[0]["content"].startswith("Completed: no\nPriority:")
    assert "Completed on" not in rows[0]["content"]


def test_upserts_always_come_before_tombstones():
    fake = _account(tasks=1)
    state: dict = {}
    _sync(fake, state)
    fake.edit_task("1", content="Edited and done")
    fake.delete_task("1")

    rows, _ = _sync(fake, state)

    kinds = ["deleted" if row["_deleted"] else "row" for row in rows]
    assert kinds == sorted(kinds), "all upserts first, then tombstones"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def test_a_selection_renders_only_the_selected_projects_tasks_and_projects():
    fake = _account(tasks=1)
    fake.add_project("P2", name="Other")
    fake.add_task("3", "P2", content="Task 3")
    state: dict = {}

    rows, stats = _sync(fake, state, project_ids=["P1"])

    assert sorted(row["id"] for row in rows) == ["project:P1", "task:1"]
    assert state["selection"] == ["P1"]
    assert state["tasks"] == {"1": "P1"}
    assert state["projects"] == ["P1"]
    assert stats["requests"] == 2, "task 3 was never read"


def test_narrowing_the_selection_forgets_what_left_it_without_asking_todoist():
    fake = _account(tasks=1)
    fake.add_project("P2")
    fake.add_task("3", "P2", content="Task 3")
    state: dict = {}
    _sync(fake, state)
    calls_before = list(fake.calls)

    rows, _ = _sync(fake, state, project_ids=["P2"])

    assert _tombstones(rows) == ["project:P1", "task:1"]
    assert ("comments", "1") not in [call for call in fake.calls if call not in calls_before]
    assert state["tasks"] == {"3": "P2"}
    assert state["projects"] == ["P2"]


def test_widening_the_selection_recovers_the_new_projects_tasks_in_a_full_resync():
    fake = _account(tasks=1)
    fake.add_project("P2")
    fake.add_task("3", "P2", content="Task 3")
    state: dict = {}
    _sync(fake, state, project_ids=["P1"])

    rows, _ = _sync(fake, state, project_ids=["P1", "P2"])

    # The selection changed, so the cursor restarts from "*": the run re-reads the live
    # account. Everything selected is emitted again — upserts under the same ids are
    # idempotent — and the corpus settles on both projects.
    assert [row["id"] for row in rows] == ["project:P1", "project:P2", "task:1", "task:3"]
    assert state["tasks"] == {"1": "P1", "3": "P2"}
    assert state["projects"] == ["P1", "P2"]
    assert state["selection"] == ["P1", "P2"]


# ---------------------------------------------------------------------------
# The budget and failures
# ---------------------------------------------------------------------------
def test_a_spent_budget_stops_the_run_cleanly_and_advances_nothing():
    fake = _account(tasks=3)
    state: dict = {}

    rows, stats = _sync(fake, state, max_requests=2)

    # sync (1) + one task's comments (1): the second task's request is not affordable.
    # Project rows cost no request, so the one of the changed project is already out.
    assert [row["id"] for row in rows] == ["project:P1", "task:1"]
    assert stats["failed_budget"] == 1
    assert "sync_token" not in state, "the cursor must not skip unrendered changes"
    assert state == {}

    rows, stats = _sync(fake, state, max_requests=DEFAULT_MAX_REQUESTS)
    assert sorted(row["id"] for row in rows) == ["project:P1", "task:1", "task:2", "task:3"]
    assert "failed_budget" not in stats
    assert state["sync_token"] == str(len(fake.journal))


def test_a_repeated_cut_run_repeats_only_idempotent_upserts():
    fake = _account(tasks=1)
    state: dict = {}

    rows, _ = _sync(fake, state, max_requests=1)  # the sync alone fits, nothing else
    assert [row["id"] for row in rows] == ["project:P1"]
    assert state == {}

    rows, _ = _sync(fake, state, max_requests=2)
    assert [row["id"] for row in rows] == ["project:P1", "task:1"]
    rows, _ = _sync(fake, state, max_requests=1000)
    assert rows == [], "the completed run committed its cursor; the next one is quiet"


def test_a_rejected_token_raises_and_keeps_the_state():
    fake = _account()
    state: dict = {}

    def reject(*args, **kwargs):
        raise TodoistAuthError(401, "/sync/v9/sync")

    fake.sync = reject
    with pytest.raises(TodoistAuthError):
        _sync(fake, state)

    assert state == {}
    assert fake.calls == []


def test_a_malformed_sync_answer_raises_and_keeps_the_state():
    class BrokenSync:
        calls = []

        def sync(self, *args, **kwargs):
            self.calls.append(("sync", args[0]))
            return {"full_sync": True}  # no sync_token

        def get_task(self, task_id, **kwargs):
            raise AssertionError("no task read happens")

        def list_comments(self, task_id, **kwargs):
            raise AssertionError("no comment read happens")

    service = BrokenSync()
    state: dict = {}
    stats = new_stats()

    with pytest.raises(TodoistResponseError, match="no sync_token"):
        list(_run_rows(service, state, stats, project_ids=None, max_requests=DEFAULT_MAX_REQUESTS))

    assert state == {}
    assert stats["failed"] == 1


def test_a_sync_answer_of_the_wrong_shape_raises_instead_of_reading_an_empty_feed():
    class ListlessSync:
        def sync(self, *args, **kwargs):
            return {"sync_token": "1", "items": "not-a-list", "projects": [], "notes": []}

        def get_task(self, task_id, **kwargs):
            raise AssertionError("no task read happens")

        def list_comments(self, task_id, **kwargs):
            raise AssertionError("no comment read happens")

    with pytest.raises(TodoistResponseError, match="items is not a list"):
        _sync(ListlessSync(), {})


def test_check_active_stops_the_run_as_it_raises():
    fake = _account()

    def revoked():
        raise RuntimeError("connection revoked")

    stats = new_stats()
    with pytest.raises(RuntimeError, match="revoked"):
        list(
            _run_rows(
                fake,
                {},
                stats,
                project_ids=None,
                max_requests=DEFAULT_MAX_REQUESTS,
                check_active=revoked,
            )
        )


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------
def test_the_factory_refuses_a_missing_token_a_bad_name_a_bad_selection_and_no_budget():
    with pytest.raises(ValueError, match="resource_name"):
        todoist_source(resource_name=" ")
    with pytest.raises(ValueError, match="_dlt_"):
        todoist_source(resource_name="_dlt_loads", access_token="t")
    with pytest.raises(ValueError, match="service or an access_token"):
        todoist_source(resource_name="r")
    with pytest.raises(ValueError, match="positive int"):
        todoist_source(resource_name="r", access_token="t", max_requests=0)
    with pytest.raises(ValueError, match="project_ids"):
        todoist_source(resource_name="r", access_token="t", project_ids="P1")
    with pytest.raises(ValueError, match="project_ids"):
        todoist_source(resource_name="r", access_token="t", project_ids=[])
    with pytest.raises(ValueError, match="project_ids"):
        todoist_source(resource_name="r", access_token="t", project_ids=["bad id"])


def test_the_built_resource_carries_the_document_markers_and_empty_stats():
    source = todoist_source(resource_name="todoist_me", service=FakeTodoist())

    assert source.name == "todoist_me"
    assert source.cognee_document_source == "todoist"
    assert source.cognee_pipeline_scope == "todoist_me"
    assert source.cognee_sync_stats == {}


# ---------------------------------------------------------------------------
# Through a real dlt pipeline
# ---------------------------------------------------------------------------
def _run(tmp_path, source) -> dict[str, str]:
    """Run ``source`` through a real dlt pipeline into SQLite; return staged id -> content."""
    import dlt

    pipeline = dlt.pipeline(
        pipeline_name="todoist_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 't.db'}"),
        dataset_name="todoist",
        pipelines_dir=str(tmp_path / "pipelines"),
    )
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id, content FROM "{source.name}" ORDER BY id'
    with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
        return {row[0]: row[1] for row in cursor.fetchall()}


def test_edits_and_deletes_survive_between_real_dlt_runs(tmp_path):
    fake = _account(tasks=2)
    fake.add_comment("1", "Keep me")
    first = todoist_source(resource_name="todoist_account", service=fake)
    assert list(_run(tmp_path, first)) == ["project:P1", "task:1", "task:2"]
    assert first.cognee_sync_stats["failed"] == 0

    quiet = todoist_source(resource_name="todoist_account", service=fake)
    staged = _run(tmp_path, quiet)
    assert list(staged) == ["project:P1", "task:1", "task:2"]
    assert quiet.cognee_sync_stats["rendered"] == 0
    assert quiet.cognee_sync_stats["requests"] == 1

    fake.edit_task("1", description="Task 1, edited")
    fake.delete_task("2")
    again = todoist_source(resource_name="todoist_account", service=fake)
    staged = _run(tmp_path, again)
    assert list(staged) == ["project:P1", "task:1"]
    assert "Task 1, edited" in staged["task:1"]
    assert "- Keep me" in staged["task:1"]


def test_state_survives_between_real_dlt_runs(tmp_path):
    fake = _account(tasks=1)
    first = todoist_source(resource_name="todoist_account", service=fake)
    _run(tmp_path, first)
    token_after_first = copy.deepcopy(first.cognee_sync_stats)

    fake.edit_task("1", content="Changed elsewhere")
    second = todoist_source(resource_name="todoist_account", service=fake)
    _run(tmp_path, second)

    assert second.cognee_sync_stats["rendered"] == 1
    assert second.cognee_sync_stats["requests"] == 2
    assert token_after_first != second.cognee_sync_stats
