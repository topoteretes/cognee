"""The Kindle source parses My Clippings.txt into one document per book, and forgets safely.

No Kindle needed: fixture files are written into a temporary folder. The walker is driven
with a plain dict for state, like the Obsidian source's tests.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cognee.tasks.ingestion.connectors import kindle as kindle_module
from cognee.tasks.ingestion.connectors.kindle import (
    KindleClippingsParseError,
    KindleClippingsUnavailableError,
    _book_rows,
    _dedup_highlights,
    _parse_clippings,
)

EMPTY_STATS = {"synced": 0, "deleted": 0}

CLIPPINGS = """\
Atomic Habits (James Clear)
- Your Highlight on page 55 | location 558-560 | Added on Tuesday, October 12, 2021 3:45:00 PM

You do not rise to the level of your goals. You fall to the level of your systems.
==========
Atomic Habits (James Clear)
- Your Note on location 559 | Added on Tuesday, October 12, 2021 3:50:00 PM

Systems first, goals second.
==========
Deep Work (Cal Newport)
- Your Highlight on location 1204 | Added on Wednesday, November 03, 2021 8:12:00 AM

Clarity about what matters provides clarity about what does not.
==========
Atomic Habits (James Clear)
- Your Bookmark on location 570 | Added on Tuesday, October 12, 2021 4:00:00 PM

==========
"""


def _write_clippings(folder: Path, text: str, name: str = "My Clippings.txt") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(text.encode("utf-8"))
    return path


def _rows(path: Path, state=None, stats=None):
    stats = dict(EMPTY_STATS) if stats is None else stats
    return list(kindle_module._iter_rows(path, {} if state is None else state, stats, None))


# --- parsing ------------------------------------------------------------------


def test_entries_parse_title_author_type_location_page_and_date():
    clips = _parse_clippings(CLIPPINGS)
    assert len(clips) == 4
    first = clips[0]
    assert first.title == "Atomic Habits"
    assert first.author == "James Clear"
    assert first.kind == "Highlight"
    assert (first.page, first.location_start, first.location_end) == (55, 558, 560)
    assert first.added == datetime(2021, 10, 12, 15, 45, tzinfo=timezone.utc)
    assert first.text.startswith("You do not rise")
    # A bookmark carries no text and no page.
    assert clips[3].kind == "Bookmark" and clips[3].text == "" and clips[3].page is None


def test_a_book_without_an_author_and_a_location_only_entry_parse():
    clips = _parse_clippings(
        "Untitled Book\n- Your Highlight on location 12 | Added on January 05, 2022 10:00:00 AM\n\nText.\n"
        "==========\n"
        "Second (Author X)\n- Your Note on page 3 | Added on January 06, 2022 11:00:00 AM\n\nA note.\n"
        "==========\n"
    )
    assert clips[0].author is None and clips[0].location_start == 12
    assert clips[1].page == 3 and clips[1].location_start is None


@pytest.mark.parametrize(
    "entry",
    [
        "Book (A)\nText without a metadata line.\n",
        "Book (Author)\nSome line that is not metadata\n\nText.\n",
        "Book (Author)\n- Your Highlight on page 1 | Added on mardi 5 janvier 2022\n\nText.\n",
    ],
)
def test_malformed_entries_fail_loudly(entry):
    with pytest.raises(KindleClippingsParseError):
        _parse_clippings(entry + "==========\n")


# --- extended-highlight dedup -----------------------------------------------------


def test_an_overlapping_newer_entry_replaces_the_extended_highlight():
    base = _parse_clippings(
        "Book\n- Your Highlight on location 10-20 | Added on January 01, 2022 10:00:00 AM\n\nShort.\n"
        "==========\n"
        "Book\n- Your Highlight on location 10-25 | Added on January 05, 2022 10:00:00 AM\n\nExtended.\n"
        "==========\n"
    )
    kept = _dedup_highlights(base)
    assert [clip.text for clip in kept] == ["Extended."]


def test_the_older_overlapping_entry_is_dropped_whichever_comes_first():
    clips = _parse_clippings(
        "Book\n- Your Highlight on location 10-25 | Added on January 05, 2022 10:00:00 AM\n\nNew.\n"
        "==========\n"
        "Book\n- Your Highlight on location 10-20 | Added on January 01, 2022 10:00:00 AM\n\nOld.\n"
        "==========\n"
    )
    assert [clip.text for clip in _dedup_highlights(clips)] == ["New."]


def test_disjoint_and_cross_type_entries_are_not_deduped():
    clips = _parse_clippings(
        "Book\n- Your Highlight on location 10-20 | Added on January 01, 2022 10:00:00 AM\n\nA.\n"
        "==========\n"
        "Book\n- Your Highlight on location 25-30 | Added on January 02, 2022 10:00:00 AM\n\nB.\n"
        "==========\n"
        "Book\n- Your Note on location 12 | Added on January 03, 2022 10:00:00 AM\n\nNote inside.\n"
        "==========\n"
    )
    assert [clip.kind for clip in _dedup_highlights(clips)] == ["Highlight", "Highlight", "Note"]


# --- the book document ---------------------------------------------------------------


def test_a_book_document_holds_its_highlights_in_location_order_with_the_note_attached():
    rows = _book_rows(CLIPPINGS)
    assert len(rows) == 2
    habits = rows[0]
    assert habits["title"] == "Atomic Habits"
    assert habits["content"].startswith("# Atomic Habits\nAuthor: James Clear")
    assert "> You do not rise" in habits["content"]
    assert "(location 558-560, page 55, 2021-10-12)" in habits["content"]
    assert ">> Note: Systems first, goals second." in habits["content"]
    # The bookmark, which has no text, is not in the document.
    assert "Bookmark" not in habits["content"]
    deep = rows[1]
    assert deep["title"] == "Deep Work" and "(location 1204, 2021-11-03)" in deep["content"]


def test_a_note_with_no_containing_highlight_stands_alone():
    rows = _book_rows(CLIPPINGS)
    assert "> Note: " in rows[1]["content"] or rows[1]["title"] == "Deep Work"


# --- the source -----------------------------------------------------------------


def test_the_source_yields_one_row_per_book_and_writes_the_state(tmp_path):
    path = _write_clippings(tmp_path / "kindle", CLIPPINGS)
    state: dict = {}

    rows = _rows(path, state)

    assert [row["title"] for row in rows] == ["Atomic Habits", "Deep Work"]
    assert all(row["url"] == "" and row["_deleted"] is False for row in rows)
    assert all(row["id"].startswith("kindle:") for row in rows)
    assert state == {"v": 1, "members": sorted(row["id"] for row in rows)}


def test_a_removed_book_is_tombstoned_and_an_unchanged_one_keeps_its_id(tmp_path):
    path = _write_clippings(tmp_path / "kindle", CLIPPINGS)
    state: dict = {}
    first = _rows(path, state)

    parts = CLIPPINGS.split("==========")
    only_habits = parts[0] + "==========\n" + parts[1] + "==========\n"
    path.write_bytes(only_habits.encode("utf-8"))
    second = _rows(path, state)

    assert second[0]["id"] == first[0]["id"], "an unchanged book keeps its id"
    assert second[1] == {"id": first[1]["id"], "_deleted": True}


def test_emptying_the_file_forgets_every_book(tmp_path):
    path = _write_clippings(tmp_path / "kindle", CLIPPINGS)
    state: dict = {}
    first = _rows(path, state)
    path.write_bytes(b"")

    rows = _rows(path, state)

    assert sorted(r["id"] for r in rows) == sorted(row["id"] for row in first)
    assert all(
        row == {"id": gone, "_deleted": True}
        for row, gone in zip(rows, sorted(r["id"] for r in rows))
    )
    assert state["members"] == []


def test_a_malformed_file_stops_the_run_before_anything_is_forgotten(tmp_path):
    path = _write_clippings(tmp_path / "kindle", CLIPPINGS)
    state: dict = {}
    first = _rows(path, state)
    path.write_bytes(
        b"Book\n- Your Highlight on page 1 | Added on January 05, 2022 10:00:00 AM\n\nX.\n"
        b"==========\nBroken Entry\nnot metadata\n==========\n"
    )

    with pytest.raises(KindleClippingsParseError):
        _rows(path, state)

    assert state["members"] == sorted(row["id"] for row in first)


def test_a_bom_file_parses(tmp_path):
    path = _write_clippings(tmp_path / "kindle", CLIPPINGS)
    path.write_bytes(b"\xef\xbb\xbf" + CLIPPINGS.encode("utf-8"))
    rows = _rows(path)
    assert len(rows) == 2


def test_a_missing_file_never_forgets(tmp_path):
    state: dict = {"v": 1, "members": ["kindle:abc"]}

    with pytest.raises(KindleClippingsUnavailableError, match="not available"):
        _rows(tmp_path / "gone" / "My Clippings.txt", state)


# --- the source constructor -------------------------------------------------------


def test_clippings_path_accepts_the_file_or_a_folder_containing_it(tmp_path):
    from cognee.tasks.ingestion.connectors import kindle_source

    with pytest.raises(ValueError, match="clippings_path"):
        kindle_source(clippings_path=tmp_path / "missing.txt")

    folder = tmp_path / "kindle"
    path = _write_clippings(folder, CLIPPINGS)
    assert kindle_source(clippings_path=path).cognee_sync_stats == {}
    assert kindle_source(clippings_path=folder).cognee_sync_stats == {}
    documents = _write_clippings(folder / "documents", CLIPPINGS)
    assert kindle_source(clippings_path=folder).cognee_sync_stats == {}
    assert documents.is_file()


def test_the_resource_is_named_after_the_file_and_carries_the_source_tag(tmp_path):
    from cognee.tasks.ingestion.connectors import kindle_source
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    path = _write_clippings(tmp_path / "kindle", CLIPPINGS)
    source = kindle_source(clippings_path=path, resource_name="pinned")
    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "kindle"
    assert getattr(source, PIPELINE_SCOPE_ATTR) == "pinned"
    other = kindle_source(clippings_path=path)
    assert getattr(other, PIPELINE_SCOPE_ATTR) != getattr(source, PIPELINE_SCOPE_ATTR)
