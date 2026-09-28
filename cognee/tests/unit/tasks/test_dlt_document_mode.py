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

from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_MAX_NAME_LENGTH,
    NODE_SET_MAX_NAMES_PER_ROW,
    STRUCTURE_MAX_KEYS,
    document_source_tag,
    is_dlt_sourced,
)
from cognee.tasks.ingestion.resolve_dlt_sources import (
    _build_document_data_item,
    _validate_row_node_set,
    _validate_row_structure,
)


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


def test_golden_drive_shaped_row_is_unaffected_by_the_row_contract():
    """A Google Drive row (id/title/content/url, see connectors/google_drive.py)
    with no reserved columns must build the exact same DataItem as before the
    node_set/structure contract existed."""
    row = SimpleNamespace(
        table_name="drive_folder_root",
        row_data={
            "id": "fileA",
            "title": "Q3 Plan",
            "content": "Ship the thing.",
            "url": "https://drive/fileA",
        },
        content_hash="h1",
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
        "url": "https://drive/fileA",
        "external_id": "fileA",
    }


def test_golden_gmail_shaped_row_is_unaffected_by_the_row_contract():
    """A Gmail row (id/title/content, no url, see connectors/gmail.py:parse_message)
    with no reserved columns must build the exact same DataItem as before."""
    row = SimpleNamespace(
        table_name="messages",
        row_data={
            "id": "msg1",
            "title": "Re: standup",
            "content": "From: a@b.com\n\nRunning late.",
        },
        content_hash="h2",
    )
    data_id = uuid5(NAMESPACE_OID, "msg1")

    item = _build_document_data_item(row, data_id, "gmail")

    assert item.data == "# Re: standup\n\nFrom: a@b.com\n\nRunning late."
    assert item.label == "Re: standup"
    assert item.data_id == data_id
    assert item.node_set is None
    assert "url" not in item.system_metadata
    assert item.system_metadata == {
        "source": "gmail",
        "title": "Re: standup",
        "table_name": "messages",
        "external_id": "msg1",
    }


def test_build_document_data_item_carries_node_set_and_structure():
    row = SimpleNamespace(
        table_name="notion_pages",
        row_data={"id": "p1", "title": "T", "content": "C"},
        content_hash="h",
    )
    item = _build_document_data_item(
        row,
        uuid4(),
        "notion",
        node_set=["notion:ws:root"],
        structure={"depth": 2},
    )
    assert item.node_set == ["notion:ws:root"]
    assert item.system_metadata["structure"] == {"depth": 2}
    # structure never overwrites the fixed keys, it only ever adds "structure".
    assert item.system_metadata["source"] == "notion"
    assert item.system_metadata["title"] == "T"


def test_build_document_data_item_without_reserved_values_omits_them():
    row = SimpleNamespace(
        table_name="notion_pages",
        row_data={"id": "p1", "title": "T", "content": "C"},
        content_hash="h",
    )
    item = _build_document_data_item(row, uuid4(), "notion")
    assert item.node_set is None
    assert "structure" not in item.system_metadata


class TestValidateRowNodeSet:
    def test_absent_value_is_not_an_issue(self):
        assert _validate_row_node_set(None, "notion") == (None, None)

    def test_already_parsed_list_postgres_shape(self):
        names, issue = _validate_row_node_set(["notion:a", "notion:b"], "notion")
        assert names == ["notion:a", "notion:b"]
        assert issue is None

    def test_json_encoded_list_sqlite_shape(self):
        names, issue = _validate_row_node_set(json.dumps(["notion:a"]), "notion")
        assert names == ["notion:a"]
        assert issue is None

    def test_single_bare_name_text_column_shape(self):
        names, issue = _validate_row_node_set("notion:solo", "notion")
        assert names == ["notion:solo"]
        assert issue is None

    def test_bad_names_are_dropped_but_good_names_and_the_row_survive(self):
        names, issue = _validate_row_node_set(["bad,name", "unprefixed", "notion:ok", ""], "notion")
        assert names == ["notion:ok"]
        assert issue == "node_set_value"

    def test_name_missing_the_source_tag_prefix_is_dropped(self):
        names, issue = _validate_row_node_set(["ws:root"], "acme")
        assert names is None
        assert issue == "node_set_value"

    def test_wrong_column_type_drops_the_whole_value(self):
        names, issue = _validate_row_node_set(123, "notion")
        assert names is None
        assert issue == "node_set_column_invalid"

    def test_names_deduped_by_graph_node_id_key(self):
        # "Notion:A" and "notion:a" collapse to the same NodeSet graph id.
        names, issue = _validate_row_node_set(["notion:a", "notion:A"], "notion")
        assert names == ["notion:a"]
        assert issue == "node_set_value"

    def test_overlong_name_is_dropped(self):
        long_name = "notion:" + "x" * NODE_SET_MAX_NAME_LENGTH
        names, issue = _validate_row_node_set([long_name], "notion")
        assert names is None
        assert issue == "node_set_value"

    def test_name_count_is_capped(self):
        many = [f"notion:{i}" for i in range(NODE_SET_MAX_NAMES_PER_ROW + 5)]
        names, issue = _validate_row_node_set(many, "notion")
        assert len(names) == NODE_SET_MAX_NAMES_PER_ROW
        assert issue == "node_set_value"


class TestValidateRowStructure:
    def test_absent_value_is_not_an_issue(self):
        assert _validate_row_structure(None) == (None, None)

    def test_already_parsed_dict_postgres_shape(self):
        structure, issue = _validate_row_structure({"a": 1, "b": "x"})
        assert structure == {"a": 1, "b": "x"}
        assert issue is None

    def test_json_encoded_dict_sqlite_shape(self):
        structure, issue = _validate_row_structure(json.dumps({"a": 1}))
        assert structure == {"a": 1}
        assert issue is None

    def test_non_dict_json_string_is_dropped(self):
        structure, issue = _validate_row_structure(json.dumps(["a", "b"]))
        assert structure is None
        assert issue == "structure_value"

    def test_unparseable_string_is_dropped(self):
        structure, issue = _validate_row_structure("not json{")
        assert structure is None
        assert issue == "structure_value"

    def test_nested_value_invalidates_the_whole_structure(self):
        structure, issue = _validate_row_structure({"a": 1, "b": [1, 2]})
        assert structure is None
        assert issue == "structure_value"

    def test_too_many_keys_is_dropped(self):
        oversized = {str(i): i for i in range(STRUCTURE_MAX_KEYS + 1)}
        structure, issue = _validate_row_structure(oversized)
        assert structure is None
        assert issue == "structure_value"

    def test_wrong_column_type_is_dropped(self):
        structure, issue = _validate_row_structure(["not", "a", "dict"])
        assert structure is None
        assert issue == "structure_value"
