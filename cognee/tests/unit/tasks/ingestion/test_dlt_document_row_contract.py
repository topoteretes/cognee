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


def test_non_finite_floats_drop_the_structure():
    from cognee.tasks.ingestion.resolve_dlt_sources import _validate_row_structure

    for raw in ('{"a": NaN}', '{"a": Infinity}', '{"a": 1e999}', {"a": float("nan")}):
        assert _validate_row_structure(raw) == (None, "structure_value")


def test_deeply_nested_json_is_dropped_not_raised():
    from cognee.tasks.ingestion.resolve_dlt_sources import (
        _validate_row_node_set,
        _validate_row_structure,
    )

    deep = "[" * 200000 + "]" * 200000
    assert _validate_row_structure(deep) == (None, "structure_value")
    assert _validate_row_node_set(deep, "notion") == (None, "node_set_value")


def test_control_characters_drop_the_name():
    from cognee.tasks.ingestion.resolve_dlt_sources import _validate_row_node_set

    names, issue = _validate_row_node_set(["notion:a\x00b", "notion:c\x07", "notion:ok"], "notion")
    assert names == ["notion:ok"]
    assert issue == "node_set_value"


def test_invisible_and_unencodable_characters_drop_the_name():
    from cognee.tasks.ingestion.resolve_dlt_sources import _validate_row_node_set

    bad = [
        "notion:\ud800",
        "notion:a\u200bb",
        "notion:\u202ex",
        "notion:a\u2028b",
        "notion:\ufeffx",
    ]
    names, issue = _validate_row_node_set(bad + ["notion:ok"], "notion")
    assert names == ["notion:ok"]
    assert issue == "node_set_value"
    assert _validate_row_node_set('["notion:\\ud800x"]', "notion") == (None, "node_set_value")


def test_a_name_empty_after_the_prefix_is_dropped():
    from cognee.tasks.ingestion.resolve_dlt_sources import _validate_row_node_set

    assert _validate_row_node_set(["notion:", "notion:   ", "notion:\xa0"], "notion") == (
        None,
        "node_set_value",
    )


def test_joiners_between_visible_characters_are_kept():
    from cognee.tasks.ingestion.resolve_dlt_sources import _validate_row_node_set

    persian = "notion:\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"
    family = "notion:\U0001f468\u200d\U0001f469\u200d\U0001f467"
    names, issue = _validate_row_node_set([persian, family], "notion")
    assert names == [persian, family]
    assert issue is None


def test_joiners_at_the_edges_or_next_to_spaces_drop_the_name():
    from cognee.tasks.ingestion.resolve_dlt_sources import _validate_row_node_set

    bad = ["notion:\u200dx", "notion:x\u200c", "notion:a \u200c b", "notion:a\u200c\u200cb"]
    assert _validate_row_node_set(bad, "notion") == (None, "node_set_value")
