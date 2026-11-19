"""Unit tests for the Letterboxd dlt source.

No network: the export is a folder or ZIP of CSV fixtures written to ``tmp_path``.
These tests exercise ``_iter_rows``/``letterboxd_source`` directly, the same
shape as ``test_obsidian_source.py``.
"""

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pytest

from cognee.tasks.ingestion.connectors import letterboxd as letterboxd_module
from cognee.tasks.ingestion.connectors import letterboxd_source
from cognee.tasks.ingestion.connectors.letterboxd import (
    LetterboxdExportUnavailableError,
    _iter_rows,
    _source_id,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_COLUMN_HINTS,
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)

TABLE = "letterboxd_test"
VERSION_GUARD = (
    "Letterboxd sync requires a Cognee build that reads per-row node sets and "
    "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
)
PLAIN_COLUMNS = {"id", "title", "content", "url", "_deleted"}
EMPTY_STATS = {
    "films_scanned": 0,
    "films_changed": 0,
    "films_unchanged": 0,
    "lists_scanned": 0,
    "lists_changed": 0,
    "deleted": 0,
}

GODFATHER_URI = "https://letterboxd.com/film/the-godfather/"
PARASITE_URI = "https://letterboxd.com/film/parasite/"

EXPORT = {
    "watched.csv": (
        "Watched Date,Film,Year,Letterboxd URI,Rating,Rewatch\n"
        "2024-01-15,The Godfather,1972," + GODFATHER_URI + ",5,\n"
        "2024-03-02,Parasite,2019," + PARASITE_URI + ",4,Yes\n"
    ),
    "ratings.csv": (
        "Date,Film,Year,Letterboxd URI,Rating\n"
        "2024-01-15,The Godfather,1972," + GODFATHER_URI + ",5\n"
    ),
    "reviews.csv": (
        "Date,Film,Year,Letterboxd URI,Rating,Review,Spoiler,Rewatch\n"
        "2024-01-15,The Godfather,1972," + GODFATHER_URI + ",5,A masterpiece.,No,\n"
    ),
    "diary.csv": (
        "Date,Film,Year,Letterboxd URI,Rating,Rewatch\n"
        "2024-01-15,The Godfather,1972," + GODFATHER_URI + ",5,\n"
    ),
    "lists/my-favorites.csv": (
        "List Name,Film,Year,Letterboxd URI\n"
        "My Favorites,The Godfather,1972," + GODFATHER_URI + "\n"
        "My Favorites,Parasite,2019," + PARASITE_URI + "\n"
    ),
}

ALL_FILM_IDS = [GODFATHER_URI, PARASITE_URI]


def _write(vault: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    return vault


def _zip_export(tmp_path: Path, files: dict[str, str]) -> Path:
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for rel, text in files.items():
            archive.writestr(rel, text)
    return zip_path


@pytest.fixture
def export_folder(tmp_path) -> Path:
    return _write(tmp_path / "export", EXPORT)


def _rows(
    export_path: Path,
    state=None,
    stats=None,
    check_active=None,
) -> list[dict]:
    stats = dict(EMPTY_STATS) if stats is None else stats
    return list(_iter_rows(export_path, {} if state is None else state, stats, check_active))


def _row(export_path: Path, state=None) -> list[dict]:
    return _rows(export_path, state)


# --- factory ---------------------------------------------------------------------------------


def test_the_resource_is_a_merge_resource_tagged_for_document_ingestion(export_folder):
    resource = letterboxd_source(export_path=export_folder, resource_name=TABLE)

    assert resource.name == TABLE
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "letterboxd"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == TABLE
    hints = resource.compute_table_schema()
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True
    assert resource.cognee_sync_stats == {}


def test_an_older_core_without_structure_support_is_refused(export_folder, monkeypatch):
    monkeypatch.setattr(letterboxd_module.dlt_utils, "DOCUMENT_SYNC_VERSION", 2)

    with pytest.raises(RuntimeError) as refused:
        letterboxd_source(export_path=export_folder)

    assert str(refused.value) == VERSION_GUARD


def test_a_missing_export_path_is_refused(tmp_path):
    with pytest.raises(ValueError, match="export_path must point to an existing"):
        letterboxd_source(export_path=tmp_path / "does-not-exist.zip")


def test_the_default_name_follows_the_resolved_export_path(tmp_path):
    export_path = _write(tmp_path / "export", EXPORT).resolve()
    expected = "letterboxd_" + _source_id(export_path).split(":")[1]

    assert letterboxd_source(export_path=export_path).name == expected
    assert letterboxd_source(export_path=str(export_path) + "/").name == expected
    other = _write(tmp_path / "other", EXPORT)
    assert letterboxd_source(export_path=other).name != expected
    assert letterboxd_source(export_path=export_path, resource_name=TABLE).name == TABLE


def test_a_bad_resource_name_is_refused(export_folder):
    with pytest.raises(ValueError, match="resource_name must be a non-empty string"):
        letterboxd_source(export_path=export_folder, resource_name="")


# --- walk ------------------------------------------------------------------------------------


def test_the_walk_merges_every_file_into_one_row_per_film(export_folder):
    rows = _rows(export_folder)
    by_id = {
        row["id"]: row
        for row in rows
        if not row.get("_deleted") and not row["id"].startswith("list:")
    }

    assert set(by_id) == set(ALL_FILM_IDS)
    # The same film appears in watched, ratings, reviews and diary: one row only.
    godfather = by_id[GODFATHER_URI]
    assert godfather["title"] == "The Godfather"
    assert godfather["url"] == GODFATHER_URI


def test_a_film_row_renders_name_year_rating_review_date_and_rewatch(export_folder):
    by_id = {row["id"]: row for row in _rows(export_folder) if not row.get("_deleted")}
    godfather = by_id[GODFATHER_URI]

    assert "Title: The Godfather" in godfather["content"]
    assert "Year: 1972" in godfather["content"]
    assert f"URI: {GODFATHER_URI}" in godfather["content"]
    assert "Rating: 5" in godfather["content"]
    assert "Review: A masterpiece." in godfather["content"]
    assert "Date: 2024-01-15" in godfather["content"]


def test_a_rewatch_film_renders_its_rewatch_flag(export_folder):
    by_id = {row["id"]: row for row in _rows(export_folder) if not row.get("_deleted")}
    parasite = by_id[PARASITE_URI]

    assert parasite["title"] == "Parasite"
    assert "Rewatch: Yes" in parasite["content"]


def test_every_curated_list_becomes_one_document(export_folder):
    rows = _rows(export_folder)
    lists = [row for row in rows if row["id"].startswith("list:") and not row.get("_deleted")]

    assert len(lists) == 1
    my_list = lists[0]
    assert my_list["title"] == "My Favorites"
    assert "The Godfather" in my_list["content"]
    assert "Parasite" in my_list["content"]
    assert GODFATHER_URI in my_list["content"]
    assert PARASITE_URI in my_list["content"]


def test_every_row_carries_the_letterboxd_node_set(export_folder):
    source_id = _source_id(export_folder)
    for row in _rows(export_folder):
        assert row[NODE_SET_COLUMN] == [source_id]


def test_every_row_carries_the_export_structure(export_folder):
    source_id = _source_id(export_folder)
    for row in _rows(export_folder):
        assert not row.get("_deleted")
        assert row[STRUCTURE_COLUMN] == {
            "ancestors": [{"kind": "export", "id": source_id, "name": "Letterboxd export"}]
        }


def test_row_has_exactly_the_document_columns(export_folder):
    rows = [row for row in _rows(export_folder) if not row.get("_deleted")]
    assert rows
    for row in rows:
        assert set(row) == PLAIN_COLUMNS | {NODE_SET_COLUMN, STRUCTURE_COLUMN}


# --- incremental -----------------------------------------------------------------------------


def test_an_unchanged_re_export_emits_no_rows(export_folder):
    state: dict = {}
    first = _rows(export_folder, state)
    assert first  # everything is new on the first run
    previous = dict(state)

    second = _rows(export_folder, previous)
    assert second == []
    assert not any(row.get("_deleted") for row in second)


def test_a_changed_film_is_re_emitted(export_folder):
    state: dict = {}
    _rows(export_folder, state)
    previous = dict(state)

    # Change The Godfather's rating.
    files = dict(EXPORT)
    files["watched.csv"] = (
        "Watched Date,Film,Year,Letterboxd URI,Rating,Rewatch\n"
        "2024-01-15,The Godfather,1972," + GODFATHER_URI + ",2,\n"
        "2024-03-02,Parasite,2019," + PARASITE_URI + ",4,Yes\n"
    )
    _write(export_folder, files)

    rows = _rows(export_folder, previous)
    by_id = {row["id"]: row for row in rows if not row.get("_deleted")}
    assert GODFATHER_URI in by_id
    assert "Rating: 2" in by_id[GODFATHER_URI]["content"]


def test_a_film_removed_from_the_export_is_tombstoned(export_folder):
    state: dict = {}
    _rows(export_folder, state)
    previous = dict(state)

    # The second export drops The Godfather from every file, so it is gone.
    parasite_only = {
        "watched.csv": (
            "Watched Date,Film,Year,Letterboxd URI,Rating,Rewatch\n"
            "2024-03-02,Parasite,2019," + PARASITE_URI + ",4,Yes\n"
        ),
        "ratings.csv": "Date,Film,Year,Letterboxd URI,Rating\n",
        "reviews.csv": "Date,Film,Year,Letterboxd URI,Rating,Review,Spoiler,Rewatch\n",
        "diary.csv": "Date,Film,Year,Letterboxd URI,Rating,Rewatch\n",
        "lists/my-favorites.csv": (
            "List Name,Film,Year,Letterboxd URI\n"
            "My Favorites,Parasite,2019," + PARASITE_URI + "\n"
        ),
    }
    _write(export_folder, parasite_only)

    rows = _rows(export_folder, previous)
    tombstones = [row for row in rows if row.get("_deleted")]
    assert tombstones == [{"id": GODFATHER_URI, "_deleted": True}]


def test_a_transient_read_error_aborts_without_tombstones(export_folder, monkeypatch):
    state: dict = {}
    _rows(export_folder, state)
    previous = dict(state)

    def failing(*args, **kwargs):
        raise OSError("drive unplugged")

    monkeypatch.setattr(letterboxd_module, "_read_export_rows", failing)
    rows = []
    with pytest.raises(OSError, match="drive unplugged"):
        rows.extend(_iter_rows(export_folder, previous, {}, None))
    assert not any(row.get("_deleted") for row in rows)


# --- zip -------------------------------------------------------------------------------------


def test_a_zip_export_is_read_the_same_as_a_folder(tmp_path):
    zip_path = _zip_export(tmp_path, EXPORT)
    rows = _rows(zip_path)
    by_id = {
        row["id"]: row
        for row in rows
        if not row.get("_deleted") and not row["id"].startswith("list:")
    }

    assert set(by_id) == set(ALL_FILM_IDS)
    lists = [row for row in rows if row["id"].startswith("list:") and not row.get("_deleted")]
    assert len(lists) == 1
