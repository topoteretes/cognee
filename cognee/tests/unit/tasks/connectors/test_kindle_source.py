from pathlib import Path

import pytest

from cognee.tasks.ingestion.connectors.kindle import (
    KindleParseError,
    KindleSourceError,
    _iter_rows,
    kindle_source,
)

SAMPLE_CLIPPINGS = """Atomic Habits (James Clear)
- Your Highlight on page 15 | location 220-225 | Added on Friday, 14 October 2022 10:15:30

You do not rise to the level of your goals. You fall to the level of your systems.
==========
Atomic Habits (James Clear)
- Your Note on page 15 | location 222 | Added on Friday, 14 October 2022 10:16:00

Remember this for system architecture design.
==========
Atomic Habits (James Clear)
- Your Bookmark on page 16 | location 230 | Added on Friday, 14 October 2022 10:17:00


==========
Thinking, Fast and Slow (Daniel Kahneman)
- Your Highlight on page 50 | location 1000-1005 | Added on Saturday, 15 October 2022 12:00:00

A reliable way to make people believe in falsehoods is frequent repetition.
==========
"""

OVERLAPPING_CLIPPINGS = """Deep Work (Cal Newport)
- Your Highlight on location 50-60 | Added on Monday, 1 January 2024 10:00:00

Deep work is the ability to focus without distraction.
==========
Deep Work (Cal Newport)
- Your Note on location 55 | Added on Monday, 1 January 2024 10:01:00

Key thesis of chapter 1.
==========
Deep Work (Cal Newport)
- Your Highlight on location 50-70 | Added on Monday, 1 January 2024 10:05:00

Deep work is the ability to focus without distraction on a cognitively demanding task.
==========
"""

NO_AUTHOR_AND_STANDALONE_NOTE = """Anonymous Manuscript
- Your Highlight on location 10-15 | Added on Sunday, 2 January 2024 09:00:00

Some anonymous wisdom.
==========
Anonymous Manuscript
- Your Note on location 100 | Added on Sunday, 2 January 2024 09:30:00

A standalone note with no highlight.
==========
"""


class TestKindleConnector:
    def test_parse_highlights_notes_and_skip_bookmarks(self, tmp_path: Path):
        clippings_file = tmp_path / "My Clippings.txt"
        clippings_file.write_text(SAMPLE_CLIPPINGS, encoding="utf-8")

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        rows = list(_iter_rows(clippings_file, state, stats))

        assert len(rows) == 2
        assert stats["synced"] == 2
        assert stats["deleted"] == 0

        atomic_habits = next(r for r in rows if r["title"] == "Atomic Habits")
        assert "Atomic Habits" in atomic_habits["title"]
        assert "James Clear" in atomic_habits["content"]
        assert "You do not rise to the level of your goals" in atomic_habits["content"]
        assert "Remember this for system architecture design" in atomic_habits["content"]
        # Bookmark should not appear
        assert "Bookmark" not in atomic_habits["content"]

        kahneman = next(r for r in rows if r["title"] == "Thinking, Fast and Slow")
        assert "frequent repetition" in kahneman["content"]

    def test_overlapping_highlight_keeps_newest_and_preserves_note(self, tmp_path: Path):
        clippings_file = tmp_path / "My Clippings.txt"
        clippings_file.write_text(OVERLAPPING_CLIPPINGS, encoding="utf-8")

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        rows = list(_iter_rows(clippings_file, state, stats))

        assert len(rows) == 1
        book_row = rows[0]
        # Should contain the extended highlight text
        assert "focus without distraction on a cognitively demanding task" in book_row["content"]
        # Should preserve the note attached to the earlier overlapping highlight
        assert "Key thesis of chapter 1" in book_row["content"]

    def test_utf8_bom_and_no_author(self, tmp_path: Path):
        clippings_file = tmp_path / "My Clippings.txt"
        # Write with UTF-8 BOM
        content_with_bom = "\ufeff" + NO_AUTHOR_AND_STANDALONE_NOTE
        clippings_file.write_bytes(content_with_bom.encode("utf-8"))

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        rows = list(_iter_rows(clippings_file, state, stats))

        assert len(rows) == 1
        row = rows[0]
        assert row["title"] == "Anonymous Manuscript"
        assert "Some anonymous wisdom" in row["content"]
        assert "A standalone note with no highlight" in row["content"]

    def test_incremental_deletion_emits_tombstone(self, tmp_path: Path):
        clippings_file = tmp_path / "My Clippings.txt"
        clippings_file.write_text(SAMPLE_CLIPPINGS, encoding="utf-8")

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        initial_rows = list(_iter_rows(clippings_file, state, stats))
        assert len(initial_rows) == 2
        assert len(state["members"]) == 2

        # Second sync: remove Thinking, Fast and Slow
        reduced_clippings = (
            SAMPLE_CLIPPINGS.split("Thinking, Fast and Slow")[0].rstrip() + "\n==========\n"
        )
        clippings_file.write_text(reduced_clippings, encoding="utf-8")

        stats2 = {"synced": 0, "deleted": 0, "failed": 0}
        second_rows = list(_iter_rows(clippings_file, state, stats2))

        # 1 active book row + 1 tombstone
        assert stats2["synced"] == 1
        assert stats2["deleted"] == 1
        tombstones = [r for r in second_rows if r.get("_deleted") is True]
        assert len(tombstones) == 1
        assert len(state["members"]) == 1

    def test_empty_file_after_prior_sync_raises_error(self, tmp_path: Path):
        clippings_file = tmp_path / "My Clippings.txt"
        clippings_file.write_text(SAMPLE_CLIPPINGS, encoding="utf-8")

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        list(_iter_rows(clippings_file, state, stats))

        # Now clear the file
        clippings_file.write_text("", encoding="utf-8")
        with pytest.raises(KindleSourceError, match="empty, but earlier runs synced"):
            list(_iter_rows(clippings_file, state, stats))

    def test_malformed_entry_raises_error(self, tmp_path: Path):
        clippings_file = tmp_path / "My Clippings.txt"
        malformed = "Just a single line\n==========\n"
        clippings_file.write_text(malformed, encoding="utf-8")

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        with pytest.raises(KindleParseError):
            list(_iter_rows(clippings_file, state, stats))

    def test_source_factory_validation(self, tmp_path: Path):
        non_existent = tmp_path / "missing.txt"
        with pytest.raises(ValueError, match="clippings_path must be an existing file"):
            kindle_source(clippings_path=non_existent)

        valid_file = tmp_path / "My Clippings.txt"
        valid_file.write_text(SAMPLE_CLIPPINGS, encoding="utf-8")

        with pytest.raises(ValueError, match="resource_name must be a non-empty string"):
            kindle_source(clippings_path=valid_file, resource_name="")

        source = kindle_source(clippings_path=valid_file)
        assert source.name.startswith("kindle_")
        assert source.cognee_document_source == "kindle"
