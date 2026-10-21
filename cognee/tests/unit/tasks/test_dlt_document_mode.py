"""Unit tests for the connector-agnostic DLT document-mode seam.

A dlt source opts into the "document" ingestion path (LLM entity extraction)
by setting ``DOCUMENT_SOURCE_ATTR`` on itself; resolve_dlt_sources then tags its
rows ``system_metadata["source"] = <tag>`` (NOT "dlt"), so ``is_dlt_sourced``
returns False and classify_documents routes them to TextDocument/cognify rather
than the deterministic manifest schema path. These tests exercise that seam
with plain objects — no connector, no database, no LLM.
"""

import json
from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    document_source_tag,
    is_dlt_sourced,
)
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item, _row_node_set


def test_document_source_tag_reads_the_marker():
    src = SimpleNamespace()
    assert document_source_tag(src) is None  # no marker -> relational

    setattr(src, DOCUMENT_SOURCE_ATTR, "notion")
    assert document_source_tag(src) == "notion"

    # empty / non-string tags are ignored (treated as not opted-in)
    setattr(src, DOCUMENT_SOURCE_ATTR, "")
    assert document_source_tag(src) is None


def test_is_dlt_sourced_only_true_for_dlt_source():
    assert is_dlt_sourced({"source": "dlt"}) is True
    # A document-mode tag is NOT "dlt", so it falls through to TextDocument.
    assert is_dlt_sourced({"source": "notion"}) is False
    assert is_dlt_sourced({"source": "google_drive"}) is False
    assert is_dlt_sourced({}) is False


def test_build_document_data_item_tags_a_non_dlt_source():
    row = SimpleNamespace(
        table_name="notion_pages",
        row_data={
            "id": "p1",
            "url": "https://example.com/p1",
            "title": "My Page",
            "content": "body text",
        },
        content_hash="abc123",
    )
    data_id = uuid5(NAMESPACE_OID, "p1")

    item = _build_document_data_item(row, data_id, "notion")

    # source != "dlt" is the whole point: it routes the row through cognify.
    assert item.system_metadata["source"] == "notion"
    assert is_dlt_sourced(item.system_metadata) is False
    assert item.system_metadata["url"] == "https://example.com/p1"
    assert item.system_metadata["external_id"] == "p1"
    assert item.system_metadata["table_name"] == "notion_pages"
    assert item.data_id == data_id
    # title becomes an H1 prefixed to the content body.
    assert item.data.startswith("# My Page")
    assert "body text" in item.data
    # Document rows are always stored as literal text, titled or not.
    assert item.literal_text is True


def test_build_document_data_item_without_title_is_just_content():
    row = SimpleNamespace(
        table_name="wiki_pages",
        row_data={"id": "x", "content": "plain body"},
        content_hash="h",
    )
    item = _build_document_data_item(row, uuid5(NAMESPACE_OID, "x"), "wiki")
    assert item.data == "plain body"
    assert item.system_metadata["source"] == "wiki"
    assert item.system_metadata["title"] is None
    assert item.literal_text is True


def test_build_document_data_item_untitled_url_content_is_marked_literal():
    # An untitled row whose content is just a URL must never be fetched as one;
    # literal_text=True is what tells ingest_data to store it as plain text.
    row = SimpleNamespace(
        table_name="wiki_pages",
        row_data={"id": "y", "content": "https://example.com/x"},
        content_hash="h2",
    )
    item = _build_document_data_item(row, uuid5(NAMESPACE_OID, "y"), "wiki")
    assert item.data == "https://example.com/x"
    assert item.literal_text is True


# ---------------------------------------------------------------- per-row node_set (SDK-863)


def _row(table_name: str, row_data: dict):
    return SimpleNamespace(
        table_name=table_name,
        row_data=row_data,
        content_hash="h",
        primary_key_value=str(row_data.get("id")),
    )


def test_golden_drive_shaped_row_builds_the_same_item_as_before():
    """A Drive row (id/title/content/url) with no reserved column is unchanged."""
    row = _row(
        "drive_folder_root",
        {"id": "fileA", "title": "Q3 Plan", "content": "Ship the thing.", "url": "https://d/fileA"},
    )
    data_id = uuid5(NAMESPACE_OID, "fileA")

    item = _build_document_data_item(row, data_id, "google_drive")

    assert item.data == "# Q3 Plan\n\nShip the thing."
    assert item.label == "Q3 Plan"
    assert item.data_id == data_id
    assert item.node_set is None
    assert item.system_metadata == {
        "source": "google_drive",
        "title": "Q3 Plan",
        "table_name": "drive_folder_root",
        "url": "https://d/fileA",
        "external_id": "fileA",
    }


def test_golden_gmail_shaped_row_builds_the_same_item_as_before():
    row = _row("messages", {"id": "msg1", "title": "Re: standup", "content": "Running late."})

    item = _build_document_data_item(row, uuid5(NAMESPACE_OID, "msg1"), "gmail")

    assert item.data == "# Re: standup\n\nRunning late."
    assert item.node_set is None
    assert item.system_metadata == {
        "source": "gmail",
        "title": "Re: standup",
        "table_name": "messages",
        "external_id": "msg1",
    }


def test_row_node_set_rides_on_the_item_and_stays_out_of_the_text():
    row = _row(
        "notion_pages",
        {"id": "p1", "title": "T", "content": "C", NODE_SET_COLUMN: ["notion:ws:root"]},
    )

    item = _build_document_data_item(row, uuid4(), "notion")

    assert item.node_set == ["notion:ws:root"]
    assert item.data == "# T\n\nC"
    assert NODE_SET_COLUMN not in item.system_metadata


def _node_set(value=None, **variants):
    """The node sets read from a row whose cognee_node_set column holds ``value``."""
    row = _row("notion_pages", {"id": "p1", "title": "T", NODE_SET_COLUMN: value, **variants})
    return _row_node_set(row, "notion")


class TestRowNodeSet:
    def test_absent_column_means_no_node_set(self):
        assert _node_set(None) is None

    def test_json_column_comes_back_as_a_list(self):
        assert _node_set(["notion:a", "notion:b"]) == ["notion:a", "notion:b"]

    def test_text_column_comes_back_as_a_json_string(self):
        assert _node_set(json.dumps(["notion:a"])) == ["notion:a"]

    def test_names_are_namespaced_under_the_source_tag(self):
        """Provider data can never name one of cognee's own node sets."""
        assert _node_set(["skills", "user_context", "ws:root"]) == [
            "notion:skills",
            "notion:user_context",
            "notion:ws:root",
        ]

    def test_already_namespaced_names_are_kept_once(self):
        assert _node_set(["notion:ws:root", "ws:root", " notion:ws:root "]) == ["notion:ws:root"]

    def test_spellings_of_one_node_set_keep_the_first(self):
        """Dedupe is by node-set identity, the same normalization the graph uses."""
        assert _node_set(["Project A", "project_a", "PROJECT A"]) == ["notion:Project A"]


class TestRowNodeSetNotAList:
    """Any shape but a JSON list stops the sync and names the row; nothing is dropped."""

    @pytest.mark.parametrize(
        ("value", "found"),
        [({"not": "a list"}, "a JSON dict"), (42, "a JSON int"), ("notion:solo", "not JSON")],
    )
    def test_the_column_itself_holds_something_else(self, value, found):
        with pytest.raises(CogneeValidationError, match=found) as error:
            _node_set(value)
        assert "row 'p1'" in str(error.value) and "notion_pages" in str(error.value)

    @pytest.mark.parametrize("entry", [3, None, "", "   ", {"a": 1}])
    def test_an_entry_that_is_not_a_non_empty_string(self, entry):
        """Same rule as add(node_set=...): a bad entry is never dropped quietly."""
        with pytest.raises(CogneeValidationError, match="not a non-empty string") as error:
            _node_set(["ws:root", entry])
        assert "row 'p1'" in str(error.value)

    def test_an_entry_that_cannot_be_a_node_set_name(self):
        with pytest.raises(CogneeValidationError, match="cannot be encoded"):
            _node_set(["ws:root\ud800"])

    def test_a_value_dlt_moved_to_a_variant_column(self):
        """A bare string under the json hint lands in cognee_node_set__v_text, this one NULL."""
        with pytest.raises(CogneeValidationError, match="cognee_node_set__v_text"):
            _node_set(None, cognee_node_set__v_text="notion:solo")

    def test_an_empty_variant_column_is_fine(self):
        """Once the column exists every row has it; only a value in it is wrong."""
        assert _node_set(["ws:root"], cognee_node_set__v_text=None) == ["notion:ws:root"]
