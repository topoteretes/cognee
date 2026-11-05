"""The Obsidian source reads a vault completely, and forgets a note only on a complete listing.

Every run walks the whole selection, one row per readable ``*.md`` note with its path as id.
A previous member the listing no longer contains is tombstoned (deleted, renamed, or outside
a narrowed ``folders``). A vault that is gone, a folder that cannot be listed or a note that
cannot be read never forgets anything. The walker is driven with a plain dict for state, the
resource through a real dlt pipeline on SQLite.
"""

import hashlib
import logging
import os
import random
import shutil
from pathlib import Path

import pytest

from cognee.tasks.ingestion.connectors import obsidian as obsidian_module
from cognee.tasks.ingestion.connectors import obsidian_source
from cognee.tasks.ingestion.connectors.obsidian import (
    ObsidianListingError,
    ObsidianVaultUnavailableError,
    _folder_list,
    _iter_rows,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_COLUMN_HINTS,
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)

TABLE = "obsidian_test"
# The wording Notion gives the same guard: the columns are what the core version is needed for.
VERSION_GUARD = (
    "Obsidian sync requires a Cognee build that reads per-row node sets and "
    "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
)
PLAIN_COLUMNS = {"id", "title", "content", "url", "_deleted"}
EMPTY_STATS = {"synced": 0, "deleted": 0, "failed": 0, "failed_read": 0}
NOTES = {
    "Inbox.md": "Call Ada about the engine.\n",
    "Projects/Work/Plan.md": "---\ntitle: Q4 plan\ntags: [work]\n---\nShip the engine.\n",
    "Projects/Home.md": "Paint the fence.\n",
}
ALL_IDS = ["obsidian:Inbox.md", "obsidian:Projects/Home.md", "obsidian:Projects/Work/Plan.md"]


BAD_TAGGED_VALUES = [
    "draft: !!bool maybe",
    "draft: !!bool 3",
    "draft: !!bool",
    "when: !!timestamp abc",
    "when: !!timestamp",
    "n: !!int",
    'n: !!int ""',
    "n: !!int +",
    'x: !!float ""',
]


def _write(vault: Path, notes: dict[str, str]) -> Path:
    for rel, text in notes.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    return vault


@pytest.fixture
def vault(tmp_path) -> Path:
    return _write(tmp_path / "vault", NOTES)


def _rows(vault: Path, folders=None, state=None, stats=None, check_active=None) -> list[dict]:
    stats = dict(EMPTY_STATS) if stats is None else stats
    return list(_iter_rows(vault, folders, {} if state is None else state, stats, check_active))


def _row(vault: Path, rel: str, text: str) -> dict:
    _write(vault, {rel: text})
    (row,) = [r for r in _rows(vault) if r["id"] == f"obsidian:{rel}"]
    return row


# --- factory ---------------------------------------------------------------------------------


def test_the_resource_is_a_merge_resource_tagged_for_document_ingestion(vault):
    resource = obsidian_source(vault_path=vault, resource_name=TABLE)

    assert resource.name == TABLE
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "obsidian"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == TABLE
    hints = resource.compute_table_schema()
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True
    assert resource.cognee_sync_stats == {}


def test_an_older_core_without_structure_support_is_refused(vault, monkeypatch):
    monkeypatch.setattr(obsidian_module.dlt_utils, "DOCUMENT_SYNC_VERSION", 2)

    with pytest.raises(RuntimeError) as refused:
        obsidian_source(vault_path=vault)

    assert str(refused.value) == VERSION_GUARD


def test_a_vault_that_is_not_a_directory_is_refused(tmp_path):
    note = _write(tmp_path / "vault", {"a.md": "a"}) / "a.md"

    for path in (tmp_path / "missing", note):
        with pytest.raises(ValueError, match="vault_path must be an existing directory"):
            obsidian_source(vault_path=path)


@pytest.mark.parametrize("name", ["", "  ", 3])
def test_a_bad_resource_name_is_refused(vault, name):
    with pytest.raises(ValueError, match="resource_name must be a non-empty string"):
        obsidian_source(vault_path=vault, resource_name=name)


# A bare string would be read as one folder per letter. An empty list is neither "everything"
# nor "nothing" to a caller. A dot folder is never read, so it would forget every note, and ..
# would leave the vault.
@pytest.mark.parametrize(
    "folders",
    [
        "Projects",
        b"Projects",
        {"Projects"},
        [],
        (),
        [""],
        ["."],
        ["/Projects"],
        ["../Projects"],
        ["Projects/../Home"],
        [".obsidian"],
        ["Projects/.git"],
        ["Projects", 3],
        [None],
        # A backslash is one name on POSIX and surrounding spaces name no folder: both would
        # select nothing and forget every note.
        ["Projects\\Work"],
        [" Projects"],
        ["Projects "],
        ["Projects\n"],
    ],
)
def test_bad_folders_are_refused_when_the_resource_is_built(vault, folders):
    with pytest.raises(ValueError, match="folders"):
        obsidian_source(vault_path=vault, folders=folders)


def test_folders_are_sorted_and_without_repeats():
    assert _folder_list(["b/", "a", "./c", "a", "b"]) == ["a", "b", "c"]
    assert _folder_list(None) is None


def test_the_default_name_follows_the_resolved_vault_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    vault = _write(tmp_path / "vault", NOTES).resolve()
    # Pinned recipe: a host stores this name, so it must not drift.
    expected = "obsidian_" + hashlib.sha256(str(vault).encode()).hexdigest()[:12]

    assert obsidian_source(vault_path=vault).name == expected
    assert obsidian_source(vault_path="~/vault").name == expected
    assert obsidian_source(vault_path=str(vault) + "/").name == expected
    other = _write(tmp_path / "other", NOTES)
    assert obsidian_source(vault_path=other).name != expected
    assert obsidian_source(vault_path=vault, resource_name=TABLE).name == TABLE


def test_moving_the_vault_folder_changes_the_default_name_but_not_a_pinned_one(tmp_path):
    first = _write(tmp_path / "before", NOTES)
    moved = first.rename(tmp_path / "after")

    assert (
        obsidian_source(vault_path=moved).name
        != "obsidian_" + hashlib.sha256(str(first.resolve()).encode()).hexdigest()[:12]
    )
    assert obsidian_source(vault_path=moved, resource_name="obsidian_pinned").name == (
        "obsidian_pinned"
    )


# --- walk ------------------------------------------------------------------------------------


def test_the_walk_reads_only_visible_markdown_notes_and_no_symlinks(tmp_path):
    outside = _write(tmp_path / "outside", {"Secret.md": "outside the vault"})
    vault = _write(
        tmp_path / "vault",
        {
            "Inbox.md": "kept",
            "Projects/Plan.md": "kept",
            ".obsidian/workspace.md": "dot dir",
            ".trash/Old.md": "dot dir",
            "Projects/.git/notes.md": "nested dot dir",
            "Projects/diagram.png": "attachment",
            "Projects/readme.txt": "not markdown",
            "Projects/Plan.markdown": "not .md",
        },
    )
    try:
        os.symlink(outside, vault / "Linked", target_is_directory=True)
        os.symlink(outside / "Secret.md", vault / "Projects" / "Secret.md")
    except OSError:
        pytest.skip("symlinks are not available here")

    assert [row["id"] for row in _rows(vault)] == ["obsidian:Inbox.md", "obsidian:Projects/Plan.md"]


def test_the_order_does_not_depend_on_the_filesystem_listing(tmp_path, monkeypatch):
    # Several notes and folders per level, so a shuffle has something to reorder.
    notes = {f"{folder}/note {n}.md": "text" for folder in "abcd" for n in range(4)}
    vault = _write(tmp_path / "vault", {**notes, **{f"top {n}.md": "text" for n in range(4)}})
    unshuffled = _rows(vault)
    real_walk = os.walk
    shuffler = random.Random(7)

    def shuffled_walk(*args, **kwargs):
        for directory, subdirs, files in real_walk(*args, **kwargs):
            shuffler.shuffle(subdirs)
            shuffler.shuffle(files)
            yield directory, subdirs, files

    monkeypatch.setattr(obsidian_module.os, "walk", shuffled_walk)

    assert _rows(vault) == unshuffled
    ids = [row["id"] for row in unshuffled]
    assert ids == sorted(ids) and len(ids) == 20


def test_folders_select_their_subfolders_and_no_folder_that_merely_starts_alike(tmp_path):
    vault = _write(
        tmp_path / "vault",
        {"Work/A.md": "a", "Work/Deep/B.md": "b", "Workshop/C.md": "c", "Work.md": "d"},
    )

    assert [r["id"] for r in _rows(vault, folders=["Work"])] == [
        "obsidian:Work/A.md",
        "obsidian:Work/Deep/B.md",
    ]
    assert [r["id"] for r in _rows(vault, folders=["Work/Deep", "Workshop"])] == [
        "obsidian:Work/Deep/B.md",
        "obsidian:Workshop/C.md",
    ]


# --- row shape -------------------------------------------------------------------------------


def test_a_root_note_has_exactly_the_document_columns(vault):
    (row,) = [r for r in _rows(vault) if r["id"] == "obsidian:Inbox.md"]

    assert set(row) == PLAIN_COLUMNS
    assert row == {
        "id": "obsidian:Inbox.md",
        "title": "Inbox",
        "content": NOTES["Inbox.md"],
        "url": "",
        "_deleted": False,
    }


def test_the_content_is_the_file_as_read_frontmatter_and_line_endings_included(tmp_path):
    text = "---\r\ntitle: Plan\r\n---\r\nBody\r\n\r\n#idea inline\r\n"
    row = _row(_write(tmp_path / "vault", {}), "Plan.md", text)

    assert row["content"] == text


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("---\ntitle: Q4 plan\n---\nBody", "Q4 plan"),
        ("---\ntitle: '  Padded  '\n---\nBody", "Padded"),
        ("---\ntitle: 2024\n---\nBody", "Note"),
        ("---\ntitle:\n---\nBody", "Note"),
        ("---\ntitle: '  '\n---\nBody", "Note"),
        ("---\ntitle: [a, b]\n---\nBody", "Note"),
        ("Body", "Note"),
    ],
)
def test_the_title_is_the_frontmatter_title_or_the_file_stem(tmp_path, text, title):
    assert _row(_write(tmp_path / "vault", {}), "Note.md", text)["title"] == title


def test_the_stem_keeps_inner_dots(tmp_path):
    assert _row(_write(tmp_path / "vault", {}), "v1.2 notes.md", "x")["title"] == "v1.2 notes"


# Rule: a tag is a string entry of tags/tag (list, or one string), "#" dropped, sorted, no
# repeats. An inline #tag in the body is not read (documented gap).
@pytest.mark.parametrize(
    ("frontmatter", "node_set"),
    [
        ("tags: [work, '#idea', work]", ["obsidian:tag:idea", "obsidian:tag:work"]),
        ("tags:\n  - zeta\n  - alpha", ["obsidian:tag:alpha", "obsidian:tag:zeta"]),
        ("tag: work", ["obsidian:tag:work"]),
        ("tags: '#work'", ["obsidian:tag:work"]),
        ("tags: [work]\ntag: idea", ["obsidian:tag:idea", "obsidian:tag:work"]),
        ("tags: [2024, '', '#', work, null]", ["obsidian:tag:work"]),
        ("tags: work, idea", ["obsidian:tag:work, idea"]),
        ("tags: #work", []),
        ("tags: 2024", []),
        ("tags:", []),
        ("aliases: [work]", []),
    ],
)
def test_the_node_set_comes_from_frontmatter_tags(tmp_path, frontmatter, node_set):
    row = _row(_write(tmp_path / "vault", {}), "Note.md", f"---\n{frontmatter}\n---\n#inline Body")

    assert row.get(NODE_SET_COLUMN, []) == node_set
    assert (NODE_SET_COLUMN in row) == bool(node_set), "the column is there only when non-empty"


def test_structure_lists_the_folders_nearest_parent_first(vault):
    rows = {row["id"]: row for row in _rows(vault)}

    assert STRUCTURE_COLUMN not in rows["obsidian:Inbox.md"]
    assert rows["obsidian:Projects/Work/Plan.md"][STRUCTURE_COLUMN] == {
        "ancestors": [
            {"kind": "folder", "id": "Projects/Work", "name": "Work"},
            {"kind": "folder", "id": "Projects", "name": "Projects"},
        ]
    }
    assert rows["obsidian:Projects/Home.md"][STRUCTURE_COLUMN] == {
        "ancestors": [{"kind": "folder", "id": "Projects", "name": "Projects"}]
    }
    assert set(rows["obsidian:Projects/Work/Plan.md"]) == PLAIN_COLUMNS | {
        NODE_SET_COLUMN,
        STRUCTURE_COLUMN,
    }


@pytest.mark.parametrize(
    "text",
    [
        "---\ntitle: [\n---\nBody",
        "---\n\ttitle: x\n---\nBody",
        "---\ndate: 2024-13-45\ntitle: Dated\ntags: [work]\n---\nBody",
        "---\n- just\n- a list\n---\nBody",
        "---\ntitle: Unterminated\ntags: [work]\nBody",
        "---\n---\nBody",
        # Deeply nested flow YAML overflows the parser's stack (RecursionError).
        "---\ntags: " + "[" * 500 + "\n---\nBody",
        # PyYAML's SafeConstructor raises KeyError, AttributeError and IndexError on an
        # explicit tag with a bad value, after the title and tags above it were read.
        *(f"---\ntitle: Dated\ntags: [work]\n{line}\n---\nBody" for line in BAD_TAGGED_VALUES),
    ],
)
def test_malformed_frontmatter_costs_the_title_and_tags_but_not_the_note(tmp_path, text):
    row = _row(_write(tmp_path / "vault", {}), "Note.md", text)

    assert row["title"] == "Note"
    assert NODE_SET_COLUMN not in row
    assert row["content"] == text


# --- state and deletion, driven with a dict --------------------------------------------------


def test_state_is_written_after_the_last_row_and_not_before(vault):
    state, stats = {}, dict(EMPTY_STATS)
    rows = _iter_rows(vault, None, state, stats)

    next(rows)
    assert state == {}, "state must not advance while rows are still being yielded"
    list(rows)

    assert state == {"v": 1, "members": ALL_IDS}
    assert stats == {**EMPTY_STATS, "synced": 3}


def test_a_member_the_listing_no_longer_contains_is_tombstoned_once(vault):
    state = {}
    _rows(vault, state=state)
    (vault / "Inbox.md").unlink()
    stats = dict(EMPTY_STATS)

    rows = _rows(vault, state=state, stats=stats)

    assert rows[-1] == {"id": "obsidian:Inbox.md", "_deleted": True}
    assert [r["id"] for r in rows if not r["_deleted"]] == ALL_IDS[1:]
    assert (stats["deleted"], state["members"]) == (1, ALL_IDS[1:])
    assert [r for r in _rows(vault, state=state) if r["_deleted"]] == []


def test_a_narrowed_selection_tombstones_what_fell_outside_it(vault):
    state = {}
    _rows(vault, state=state)

    rows = _rows(vault, folders=["Projects/Work"], state=state)

    assert [r["id"] for r in rows if r["_deleted"]] == [ALL_IDS[0], ALL_IDS[1]]
    assert state["members"] == [ALL_IDS[2]]


def test_a_missing_vault_raises_before_any_row_and_leaves_state_alone(vault):
    state = {"v": 1, "members": ALL_IDS}
    shutil.rmtree(vault)

    with pytest.raises(ObsidianVaultUnavailableError) as gone:
        _rows(vault, state=state)

    assert state == {"v": 1, "members": ALL_IDS}
    assert str(vault) not in str(gone.value)


def test_a_folder_that_cannot_be_listed_raises_and_forgets_nothing(vault):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("directory permissions are not enforced here")
    state = {"v": 1, "members": ALL_IDS}
    locked = vault / "Projects" / "Work"
    locked.chmod(0o000)
    try:
        with pytest.raises(ObsidianListingError) as refused:
            _rows(vault, state=state)
    finally:
        locked.chmod(0o755)

    assert state == {"v": 1, "members": ALL_IDS}
    assert "Projects/Work" in str(refused.value) and str(vault) not in str(refused.value)


def test_an_unreadable_root_is_a_listing_error_not_an_empty_root(vault):
    """The empty-root check must come after the listing: listing the root is what turns a
    permission error into ObsidianListingError, os.listdir would raise it raw, path included."""
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("directory permissions are not enforced here")
    state = {"v": 1, "members": ALL_IDS}
    vault.chmod(0o000)
    try:
        with pytest.raises(ObsidianListingError) as refused:
            _rows(vault, state=state)
    finally:
        vault.chmod(0o755)

    assert state == {"v": 1, "members": ALL_IDS}
    assert str(vault) not in str(refused.value)


def test_an_unreadable_note_is_counted_and_stays_a_member(vault):
    state = {"v": 1, "members": ALL_IDS}
    (vault / "Inbox.md").write_bytes(b"\xff\xfe not utf-8")
    (vault / "Projects" / "New.md").write_bytes(b"\xff\xfe also not utf-8")
    stats = dict(EMPTY_STATS)

    rows = _rows(vault, state=state, stats=stats)

    assert [r["id"] for r in rows] == ALL_IDS[1:], "no row, and no tombstone, for either note"
    assert (stats["synced"], stats["deleted"], stats["failed"], stats["failed_read"]) == (
        2,
        0,
        2,
        2,
    )
    assert state["members"] == ALL_IDS, "the old member stays, the note never synced is not one"


def test_check_active_runs_before_each_read_and_its_error_is_not_a_failed_note(vault):
    calls = []

    def revoked():
        calls.append(1)
        if len(calls) == 2:
            raise PermissionError("access revoked")

    state, stats = {}, dict(EMPTY_STATS)
    with pytest.raises(PermissionError, match="access revoked"):
        _rows(vault, state=state, stats=stats, check_active=revoked)

    assert (len(calls), state, stats["failed"]) == (2, {}, 0)


# --- through a real dlt pipeline -------------------------------------------------------------


def _pipeline(tmp_path):
    import dlt

    return dlt.pipeline(
        pipeline_name="obsidian_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'a.db'}"),
        dataset_name="obsidian",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


# ingest_dlt_source runs every document source with these hints: without the json hint dlt would
# turn the node set column into a child table, and a tombstone is tested against that.
def _run(tmp_path, source, table: str = TABLE) -> dict[str, str]:
    pipeline = _pipeline(tmp_path)
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id, content FROM "{table}" ORDER BY id'
    with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
        return {row[0]: row[1] for row in cursor.fetchall()}


def _source(vault, **kwargs):
    return obsidian_source(
        vault_path=vault, resource_name=kwargs.pop("resource_name", TABLE), **kwargs
    )


def test_sync_edit_delete_and_rename_through_real_dlt_runs(tmp_path, vault):
    first = _source(vault)
    assert list(_run(tmp_path, first)) == ALL_IDS
    assert first.cognee_sync_stats == {**EMPTY_STATS, "synced": 3}

    quiet = _source(vault)
    assert _run(tmp_path, quiet) == {k: NOTES[k.removeprefix("obsidian:")] for k in ALL_IDS}
    assert quiet.cognee_sync_stats == {**EMPTY_STATS, "synced": 3}

    _write(vault, {"Inbox.md": "Call Ben instead.\n"})
    assert _run(tmp_path, _source(vault))["obsidian:Inbox.md"] == "Call Ben instead.\n"

    (vault / "Projects" / "Home.md").unlink()
    deleting = _source(vault)
    assert list(_run(tmp_path, deleting)) == [ALL_IDS[0], ALL_IDS[2]]
    assert deleting.cognee_sync_stats["deleted"] == 1

    (vault / "Inbox.md").rename(vault / "Projects" / "Moved.md")
    assert list(_run(tmp_path, _source(vault))) == [
        "obsidian:Projects/Moved.md",
        "obsidian:Projects/Work/Plan.md",
    ]


def test_narrowing_folders_forgets_the_rest_and_widening_ingests_it_again(tmp_path, vault):
    _run(tmp_path, _source(vault))

    narrowed = _source(vault, folders=["Projects/Work"])
    assert list(_run(tmp_path, narrowed)) == [ALL_IDS[2]]
    assert narrowed.cognee_sync_stats["deleted"] == 2

    assert list(_run(tmp_path, _source(vault, folders=["Projects"]))) == ALL_IDS[1:]
    assert list(_run(tmp_path, _source(vault))) == ALL_IDS


def test_a_vault_that_disappears_between_runs_forgets_nothing(tmp_path, vault):
    from dlt.pipeline.exceptions import PipelineStepFailed

    _run(tmp_path, _source(vault))
    backup = tmp_path / "backup"
    shutil.copytree(vault, backup)
    # Built while the vault is there, run after it is gone, like an unmounted drive.
    gone = _source(vault)
    shutil.rmtree(vault)
    with pytest.raises(PipelineStepFailed, match="vault folder is not available"):
        _run(tmp_path, gone)
    assert gone.cognee_sync_stats == EMPTY_STATS

    # Back on a mounted drive, minus one note: the state still holds all three members.
    backup.rename(vault)
    (vault / "Inbox.md").unlink()
    back = _source(vault)
    assert list(_run(tmp_path, back)) == ALL_IDS[1:]
    assert back.cognee_sync_stats["deleted"] == 1


def test_an_empty_vault_is_a_complete_listing_and_forgets_every_note(tmp_path, vault):
    _run(tmp_path, _source(vault))
    for rel in NOTES:
        (vault / rel).unlink()

    emptied = _source(vault)

    assert _run(tmp_path, emptied) == {}
    assert emptied.cognee_sync_stats["deleted"] == 3


def test_an_unreadable_note_keeps_its_row_through_real_runs(tmp_path, vault):
    _run(tmp_path, _source(vault))
    (vault / "Inbox.md").write_bytes(b"\xff\xfe broken")

    broken = _source(vault)
    rows = _run(tmp_path, broken)

    assert rows["obsidian:Inbox.md"] == NOTES["Inbox.md"], "the staged row is not touched"
    assert (broken.cognee_sync_stats["failed"], broken.cognee_sync_stats["deleted"]) == (1, 0)

    _write(vault, {"Inbox.md": "Fixed.\n"})
    assert _run(tmp_path, _source(vault))["obsidian:Inbox.md"] == "Fixed.\n"


def test_two_resource_names_in_one_dataset_stay_apart(tmp_path, vault):
    other = _write(tmp_path / "other", {"Other.md": "another vault"})
    _run(tmp_path, _source(vault, resource_name="obsidian_a"), "obsidian_a")
    _run(tmp_path, _source(other, resource_name="obsidian_b"), "obsidian_b")

    (vault / "Inbox.md").unlink()
    first = _run(tmp_path, _source(vault, resource_name="obsidian_a"), "obsidian_a")
    second = _run(tmp_path, _source(other, resource_name="obsidian_b"), "obsidian_b")

    assert list(first) == ALL_IDS[1:]
    assert list(second) == ["obsidian:Other.md"]


def test_a_run_that_raises_keeps_the_old_state_so_the_deletion_still_happens(tmp_path, vault):
    from dlt.pipeline.exceptions import PipelineStepFailed

    _run(tmp_path, _source(vault))
    (vault / "Inbox.md").unlink()
    calls = []

    def revoked():
        calls.append(1)
        if len(calls) > 1:
            raise PermissionError("access revoked")

    stopped = _source(vault, check_active=revoked)
    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, stopped)
    assert set(stopped.cognee_sync_stats) == set(EMPTY_STATS)

    assert list(_run(tmp_path, _source(vault))) == ALL_IDS[1:]


def test_check_active_stops_a_run_before_any_note_is_read(tmp_path, vault):
    from dlt.pipeline.exceptions import PipelineStepFailed

    def revoked():
        raise PermissionError("access revoked")

    source = _source(vault, check_active=revoked)
    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, source)

    assert source.cognee_sync_stats == EMPTY_STATS


def test_a_revoked_connection_publishes_no_tombstone(tmp_path, vault):
    """With no note left to read, only the check around the published rows can stop the run."""
    from dlt.pipeline.exceptions import PipelineStepFailed

    _run(tmp_path, _source(vault))
    for rel in NOTES:
        (vault / rel).unlink()

    def revoked():
        raise PermissionError("access revoked")

    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, _source(vault, check_active=revoked))

    assert list(_run(tmp_path, _source(vault))) == [], "the deletions wait for the next run"


def test_the_counts_of_one_run_do_not_carry_into_the_next(tmp_path, vault):
    source = _source(vault)
    _run(tmp_path, source)
    (vault / "Inbox.md").unlink()

    _run(tmp_path, source)

    assert source.cognee_sync_stats == {**EMPTY_STATS, "synced": 2, "deleted": 1}


# --- a folder that selects nothing must not forget the vault -------------------------------


def _wrong_case_exists(monkeypatch, wrong: str):
    """Imitate a case-insensitive file system: ``wrong`` exists on disk under any case."""
    real_exists = Path.exists

    def exists(path):
        return path.name == wrong or real_exists(path)

    monkeypatch.setattr(obsidian_module.Path, "exists", exists)


def test_a_folder_that_exists_under_another_case_raises_and_forgets_nothing(vault, monkeypatch):
    state = {"v": 1, "members": ALL_IDS}
    _wrong_case_exists(monkeypatch, "projects")

    with pytest.raises(ObsidianListingError) as refused:
        _rows(vault, folders=["projects"], state=state)

    assert state == {"v": 1, "members": ALL_IDS}
    assert "'projects'" in str(refused.value) and str(vault) not in str(refused.value)


def test_a_wrong_case_folder_on_a_case_insensitive_file_system_is_refused_for_real(vault):
    if (vault / "PROJECTS").exists() is False:
        pytest.skip("this file system is case sensitive")

    with pytest.raises(ObsidianListingError):
        _rows(vault, folders=["PROJECTS"])


def test_a_selected_symlinked_folder_raises_instead_of_selecting_nothing(tmp_path, vault):
    try:
        os.symlink(vault / "Projects", vault / "Linked", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available here")

    with pytest.raises(ObsidianListingError, match="Linked") as refused:
        _rows(vault, folders=["Linked"])

    # The cause is not always the letter case, so the message must not say it is.
    assert "check the letter case" not in str(refused.value)
    assert "symbolic link" in str(refused.value) and str(vault) not in str(refused.value)


def test_a_removed_selected_folder_still_forgets_its_notes(vault):
    state = {}
    _rows(vault, folders=["Projects"], state=state)
    shutil.rmtree(vault / "Projects")

    rows = _rows(vault, folders=["Projects"], state=state)

    assert [r["id"] for r in rows] == ALL_IDS[1:], "only tombstones, the folder is really gone"
    assert state["members"] == []


def test_a_wrong_case_folder_through_a_real_dlt_run_keeps_every_row(tmp_path, vault, monkeypatch):
    from dlt.pipeline.exceptions import PipelineStepFailed

    assert list(_run(tmp_path, _source(vault, folders=["Projects"]))) == ALL_IDS[1:]
    _wrong_case_exists(monkeypatch, "projects")

    wrong = _source(vault, folders=["projects"])
    with pytest.raises(PipelineStepFailed, match="exactly that name"):
        _run(tmp_path, wrong)
    assert wrong.cognee_sync_stats == EMPTY_STATS

    monkeypatch.undo()
    assert list(_run(tmp_path, _source(vault, folders=["Projects"]))) == ALL_IDS[1:]


# --- byte order mark and the published count -------------------------------------------------


def test_a_byte_order_mark_does_not_hide_the_frontmatter(tmp_path):
    text = "\ufeff---\ntitle: Q4 plan\ntags: [work]\n---\nBody\n"
    row = _row(_write(tmp_path / "vault", {}), "Note.md", text)

    assert row["title"] == "Q4 plan"
    assert row[NODE_SET_COLUMN] == ["obsidian:tag:work"]
    assert row["content"] == text, "the BOM stays in the content"


def test_a_row_stopped_before_it_was_published_is_not_counted(tmp_path, vault):
    from dlt.pipeline.exceptions import PipelineStepFailed

    calls = []

    def revoked_after_the_first_row_was_built():
        # Per row: before guarded_rows pulls, before the note is read, and once the row is
        # built and about to be published. The third call is that last one.
        calls.append(1)
        if len(calls) == 3:
            raise PermissionError("access revoked")

    source = _source(vault, check_active=revoked_after_the_first_row_was_built)
    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, source)

    assert source.cognee_sync_stats["synced"] == 0


def test_a_tombstone_stopped_before_it_was_published_is_not_counted(tmp_path, vault):
    from dlt.pipeline.exceptions import PipelineStepFailed

    _run(tmp_path, _source(vault))
    for note in [path for path in vault.rglob("*.md") if ".obsidian" not in path.parts]:
        note.unlink()
    calls = []

    def revoked_once_the_tombstone_is_built():
        # No note is read, so the calls are guarded_rows before it pulls and again once the
        # tombstone is built and about to be published. The second call is that last one.
        calls.append(1)
        if len(calls) == 2:
            raise PermissionError("access revoked")

    source = _source(vault, check_active=revoked_once_the_tombstone_is_built)
    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, source)

    assert source.cognee_sync_stats["deleted"] == 0


# The parser overflows its stack, or a constructor raises on an explicit tag with a bad value.
@pytest.mark.parametrize(
    "frontmatter",
    ["tags: " + "[" * 500, *BAD_TAGGED_VALUES],
)
def test_a_note_whose_frontmatter_the_parser_chokes_on_does_not_stop_the_run(vault, frontmatter):
    state = {}
    _rows(vault, state=state)
    _write(vault, {"Deep.md": f"---\n{frontmatter}\n---\nBody"})
    (vault / "Inbox.md").unlink()

    rows = _rows(vault, state=state)

    assert [r["id"] for r in rows] == [
        "obsidian:Deep.md",
        "obsidian:Projects/Home.md",
        "obsidian:Projects/Work/Plan.md",
        "obsidian:Inbox.md",
    ]
    assert "obsidian:Deep.md" in state["members"] and ALL_IDS[0] not in state["members"]


# --- what counts as a note --------------------------------------------------------------------


def test_a_fifo_named_like_a_note_is_never_listed_or_read(vault):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs on this platform")
    os.mkfifo(vault / "Pipe.md")

    # Listing only: reading a FIFO with no writer would block forever.
    assert obsidian_module._list_notes(vault, None) == [
        i.removeprefix("obsidian:") for i in ALL_IDS
    ]


def test_hidden_files_are_not_listed_at_the_root_or_in_a_folder(vault):
    _write(vault, {".hid.md": "hidden", "Projects/._note.md": "AppleDouble"})

    assert [r["id"] for r in _rows(vault)] == ALL_IDS


def test_a_hidden_file_that_was_ingested_before_is_tombstoned_once(vault):
    _write(vault, {".hid.md": "hidden"})
    state = {"v": 1, "members": ["obsidian:.hid.md", *ALL_IDS]}

    rows = _rows(vault, state=state)

    assert [r["id"] for r in rows if r["_deleted"]] == ["obsidian:.hid.md"]
    assert "obsidian:.hid.md" not in state["members"]
    assert [r for r in _rows(vault, state=state) if r["_deleted"]] == []


# --- a root with nothing in it ----------------------------------------------------------------


def test_an_empty_root_with_members_raises_and_forgets_nothing(tmp_path):
    empty = tmp_path / "mount"
    empty.mkdir()
    state = {"v": 1, "members": ALL_IDS}

    with pytest.raises(ObsidianVaultUnavailableError, match="did not mount") as refused:
        _rows(empty, state=state)

    assert state == {"v": 1, "members": ALL_IDS}
    assert str(empty) not in str(refused.value)


def test_an_empty_root_without_members_syncs_nothing_and_raises_nothing(tmp_path):
    empty = tmp_path / "mount"
    empty.mkdir()
    state = {}

    assert _rows(empty, state=state) == []
    assert state == {"v": 1, "members": []}


def test_a_root_holding_only_a_hidden_folder_is_a_vault_and_forgets_its_notes(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    state = {"v": 1, "members": ALL_IDS}

    rows = _rows(vault, state=state)

    assert [r["id"] for r in rows] == ALL_IDS
    assert all(r["_deleted"] for r in rows)
    assert state["members"] == []


def test_an_empty_root_through_a_real_dlt_run_keeps_every_row(tmp_path, vault):
    from dlt.pipeline.exceptions import PipelineStepFailed

    _run(tmp_path, _source(vault))
    # Built while the vault has notes, run after the drive came back empty.
    source = _source(vault)
    shutil.rmtree(vault)
    vault.mkdir()

    with pytest.raises(PipelineStepFailed, match="did not mount"):
        _run(tmp_path, source)

    # The drive comes back, minus one note: the state still holds every member.
    _write(vault, {k: v for k, v in NOTES.items() if k != "Inbox.md"})
    assert list(_run(tmp_path, _source(vault))) == ALL_IDS[1:]


# --- a selected folder that is not there ------------------------------------------------------


def _missing_folder_warnings(caplog) -> list:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "does not exist in the vault" in r.getMessage()
    ]


def test_a_folder_that_never_existed_warns_and_changes_nothing(vault, caplog):
    state = {}

    rows = _rows(vault, folders=["Projecs"], state=state)

    assert rows == [] and state["members"] == []
    (warning,) = _missing_folder_warnings(caplog)
    assert "'Projecs'" in warning.getMessage() and str(vault) not in warning.getMessage()
    # A user who removed the folder on purpose gets this on every run, so it says what to do.
    assert "remove it from folders" in warning.getMessage()


def test_a_removed_folder_with_members_forgets_them_without_a_warning(vault, caplog):
    state = {}
    _rows(vault, folders=["Projects"], state=state)
    shutil.rmtree(vault / "Projects")
    caplog.clear()

    rows = _rows(vault, folders=["Projects"], state=state)

    assert all(r["_deleted"] for r in rows) and len(rows) == 2
    assert _missing_folder_warnings(caplog) == []


def test_a_folder_that_exists_does_not_warn(vault, caplog):
    _rows(vault, folders=["Projects"])

    assert _missing_folder_warnings(caplog) == []


def test_a_file_named_in_folders_is_called_a_file_and_forgets_nothing(vault):
    state = {"v": 1, "members": ALL_IDS}

    with pytest.raises(ObsidianListingError, match="'Inbox.md' is a file, not a folder") as refused:
        _rows(vault, folders=["Inbox.md"], state=state)

    assert state == {"v": 1, "members": ALL_IDS}
    assert str(vault) not in str(refused.value)
