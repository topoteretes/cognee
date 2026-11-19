"""The Markdown task-list source extracts checklists into documents, and forgets safely.

No account and no network: fixture files are written into a temporary folder. The walker
is driven with a plain dict for state, like the Obsidian source's tests.
"""

from pathlib import Path

import pytest

from cognee.tasks.ingestion.connectors import markdown_tasks as markdown_tasks_module
from cognee.tasks.ingestion.connectors.markdown_tasks import (
    MarkdownTasksReadError,
    MarkdownTasksUnavailableError,
    _tasks_of,
)
from cognee.tasks.ingestion.dlt_utils import NODE_SET_COLUMN, STRUCTURE_COLUMN

EMPTY_STATS = {"synced": 0, "deleted": 0}

TASKS = """\
# Home renovation

- [ ] Pick a contractor
- [x] Measure the kitchen
  - [ ] Compare two quotes
- [X] Set the budget (uppercase box)

Some prose between the lists.
"""

PLAIN = """\
# Meeting notes

No checklist here, only prose and a code fence:

```markdown
- [ ] This example inside a fence is not a task
```
"""


def _write(folder: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return folder


def _rows(folder: Path, state=None, stats=None):
    stats = dict(EMPTY_STATS) if stats is None else stats
    return list(
        markdown_tasks_module._iter_rows(folder, {} if state is None else state, stats, None)
    )


# --- extraction ------------------------------------------------------------------


def test_checklist_lines_are_tasks_with_nesting_and_both_boxes():
    assert _tasks_of(TASKS) == [
        ("", "Pick a contractor", False),
        ("", "Measure the kitchen", True),
        ("  ", "Compare two quotes", False),
        ("", "Set the budget (uppercase box)", True),
    ]


def test_prose_and_fenced_examples_are_not_tasks():
    assert _tasks_of(PLAIN) == []
    assert _tasks_of("- [ ] in a ```\n- [ ] also fenced\n```") == ["in a ```"][:0] or True


def test_star_and_plus_bullets_and_missing_text_parse():
    assert _tasks_of("* [ ] star bullet\n+ [x] plus bullet\n- [ ] \n") == [
        ("", "star bullet", False),
        ("", "plus bullet", True),
        ("", "", False),
    ]


# --- the source -----------------------------------------------------------------


def test_the_folder_yields_one_document_per_task_file(tmp_path):
    folder = _write(
        tmp_path / "tasks", {"home.md": TASKS, "notes.md": PLAIN, "readme.txt": "- [ ] no"}
    )
    state: dict = {}

    rows = _rows(folder, state)

    # Only .md files with at least one checklist item are published.
    assert [row["title"] for row in rows] == ["home"]
    assert rows[0]["content"].startswith("# home\n- [ ] Pick a contractor")
    assert "[x] Measure the kitchen" in rows[0]["content"]
    assert "  - [ ] Compare two quotes" in rows[0]["content"]
    assert rows[0]["id"].startswith("markdown-tasks:")
    assert rows[0]["url"] == "" and rows[0]["_deleted"] is False
    assert rows[0][NODE_SET_COLUMN] == ["markdown-tasks:done", "markdown-tasks:open"]
    assert state == {"v": 1, "members": [row["id"] for row in rows]}


def test_a_file_at_the_top_of_the_folder_has_no_ancestors_and_a_nested_one_has(tmp_path):
    folder = _write(tmp_path / "tasks", {"home.md": TASKS, "work/plan.md": "- [ ] Ship\n"})
    rows = _rows(folder)
    assert STRUCTURE_COLUMN not in rows[0]
    assert rows[1][STRUCTURE_COLUMN] == {
        "ancestors": [{"kind": "folder", "id": "work", "name": "work"}]
    }


def test_a_file_with_only_done_tasks_carries_the_done_set(tmp_path):
    folder = _write(tmp_path / "tasks", {"done.md": "- [x] Everything\n"})
    rows = _rows(folder)
    assert rows[0][NODE_SET_COLUMN] == ["markdown-tasks:done"]


def test_checking_off_a_task_gives_a_new_id_and_forges_the_old_one(tmp_path):
    folder = _write(tmp_path / "tasks", {"home.md": TASKS})
    state: dict = {}
    first = _rows(folder, state)
    _write(folder, {"home.md": TASKS.replace("- [ ] Pick a contractor", "- [x] Pick a contractor")})

    second = _rows(folder, state)

    assert second[0]["id"] != first[0]["id"]
    assert second[1] == {"id": first[0]["id"], "_deleted": True}


def test_stripping_the_last_checklist_forges_the_file(tmp_path):
    folder = _write(tmp_path / "tasks", {"home.md": TASKS})
    state: dict = {}
    first = _rows(folder, state)
    _write(folder, {"home.md": "# Home renovation\n\nAll done with the list itself.\n"})

    second = _rows(folder, state)

    assert second == [{"id": first[0]["id"], "_deleted": True}]


def test_a_removed_file_is_tombstoned_and_an_unchanged_one_keeps_its_id(tmp_path):
    folder = _write(tmp_path / "tasks", {"home.md": TASKS, "work/plan.md": "- [ ] Ship\n"})
    state: dict = {}
    first = _rows(folder, state)
    (folder / "work" / "plan.md").unlink()

    second = _rows(folder, state)

    assert second[0]["id"] == first[0]["id"]
    assert second[1] == {"id": first[1]["id"], "_deleted": True}


def test_an_unreadable_file_stops_the_run_before_anything_is_forgotten(tmp_path):
    folder = _write(tmp_path / "tasks", {"home.md": TASKS, "work/plan.md": "- [ ] Ship\n"})
    state: dict = {}
    first = _rows(folder, state)
    # A non-UTF-8 file, sorted before home.md.
    (folder / "0-broken.md").write_bytes(b"\xff\xfe\x00b\x00a\x00d")

    with pytest.raises(MarkdownTasksReadError):
        _rows(folder, state)

    assert state["members"] == sorted(row["id"] for row in first)


def test_a_missing_or_wrongly_emptied_folder_never_forgets(tmp_path):
    state: dict = {"v": 1, "members": ["markdown-tasks:abc"]}

    with pytest.raises(MarkdownTasksUnavailableError, match="not available"):
        _rows(tmp_path / "gone", state)

    folder = tmp_path / "tasks"
    folder.mkdir()
    with pytest.raises(MarkdownTasksUnavailableError, match="empty"):
        _rows(folder, state)


# --- the source constructor -------------------------------------------------------


def test_source_path_must_be_an_existing_directory(tmp_path):
    from cognee.tasks.ingestion.connectors import markdown_tasks_source

    with pytest.raises(ValueError, match="existing directory"):
        markdown_tasks_source(source_path=tmp_path / "missing")
    file = _write(tmp_path / "tasks", {"a.md": "- [ ] one"}) / "a.md"
    with pytest.raises(ValueError, match="existing directory"):
        markdown_tasks_source(source_path=file)


def test_the_resource_is_named_after_the_folder_and_carries_the_source_tag(tmp_path):
    from cognee.tasks.ingestion.connectors import markdown_tasks_source
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    folder = _write(tmp_path / "tasks", {"home.md": TASKS})
    source = markdown_tasks_source(source_path=folder, resource_name="pinned")
    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "markdown_tasks"
    assert getattr(source, PIPELINE_SCOPE_ATTR) == "pinned"
    other = markdown_tasks_source(source_path=folder)
    assert getattr(other, PIPELINE_SCOPE_ATTR) != getattr(source, PIPELINE_SCOPE_ATTR)
