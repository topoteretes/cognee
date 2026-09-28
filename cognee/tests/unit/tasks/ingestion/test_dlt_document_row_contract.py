"""Tests for the document-mode row contract's plumbing in ingest_dlt_source.

Covers the two reserved columns' effect on row identity (content_hash) and on
child-table handling — both pure functions, no database, no dlt runtime.
"""

from cognee.tasks.ingestion.dlt_utils import NODE_SET_COLUMN, STRUCTURE_COLUMN
from cognee.tasks.ingestion.ingest_dlt_source import (
    _row_content_hash,
    _skip_child_tables_for_document_mode,
)


class TestRowContentHash:
    def test_a_null_reserved_column_does_not_change_the_hash(self):
        with_null_column = _row_content_hash(
            {"title": "t", "content": "c", NODE_SET_COLUMN: None, STRUCTURE_COLUMN: None}
        )
        without_column = _row_content_hash({"title": "t", "content": "c"})
        assert with_null_column == without_column

    def test_a_set_reserved_column_does_change_the_hash(self):
        without_column = _row_content_hash({"title": "t", "content": "c"})
        with_value = _row_content_hash({"title": "t", "content": "c", NODE_SET_COLUMN: ["acme:a"]})
        assert with_value != without_column

    def test_changing_a_set_reserved_column_changes_the_hash(self):
        first = _row_content_hash({"title": "t", NODE_SET_COLUMN: ["acme:a"]})
        second = _row_content_hash({"title": "t", NODE_SET_COLUMN: ["acme:b"]})
        assert first != second

    def test_rows_without_reserved_columns_are_unaffected(self):
        # Drive/Gmail rows never carry these columns at all — same input,
        # same hash, regardless of this feature existing.
        row = {"id": "fileA", "title": "Q3", "content": "Ship it.", "url": "https://drive/fileA"}
        assert _row_content_hash(dict(row)) == _row_content_hash(dict(row))


class TestSkipChildTablesForDocumentMode:
    def test_a_plain_table_with_no_parent_is_kept(self):
        schema_tables = {"notion_pages": {}}
        kept = _skip_child_tables_for_document_mode({"notion_pages"}, schema_tables, "notion")
        assert kept == {"notion_pages"}

    def test_a_child_table_is_dropped(self):
        schema_tables = {
            "notion_pages": {},
            "notion_pages__tags": {"parent": "notion_pages"},
        }
        kept = _skip_child_tables_for_document_mode(
            {"notion_pages", "notion_pages__tags"}, schema_tables, "notion"
        )
        assert kept == {"notion_pages"}

    def test_a_table_absent_from_the_schema_map_is_kept(self):
        # Defensive: a table dlt reports as loaded but that isn't in the
        # schema dict (shouldn't happen) must not be silently dropped.
        kept = _skip_child_tables_for_document_mode({"unknown_table"}, {}, "notion")
        assert kept == {"unknown_table"}

    def test_a_reserved_column_child_table_logs_a_hint(self, caplog):
        schema_tables = {
            "notion_pages": {},
            "notion_pages__cognee_node_set": {"parent": "notion_pages"},
        }
        with caplog.at_level("WARNING"):
            kept = _skip_child_tables_for_document_mode(
                {"notion_pages", "notion_pages__cognee_node_set"}, schema_tables, "notion"
            )
        assert kept == {"notion_pages"}
        assert "cognee_node_set" in caplog.text
        assert "data_type" in caplog.text

    def test_an_unrelated_child_table_logs_no_reserved_column_hint(self, caplog):
        schema_tables = {
            "notion_pages": {},
            "notion_pages__attachments": {"parent": "notion_pages"},
        }
        with caplog.at_level("WARNING"):
            kept = _skip_child_tables_for_document_mode(
                {"notion_pages", "notion_pages__attachments"}, schema_tables, "notion"
            )
        assert kept == {"notion_pages"}
        assert "data_type" not in caplog.text
