"""Tests for ingest_data's call-level + item-level node_set union.

ingest_data writes the union of the call's node_set param and a DataItem's
own node_set to external_metadata["node_set"] and Data.node_set. With no
item-level node_set the union must be a no-op: byte-identical to the
pre-existing call-level-only behavior, duplicates and all.
"""

import json
from uuid import uuid4

from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.tasks.documents.classify_documents import update_node_set
from cognee.tasks.ingestion.ingest_data import _union_node_sets


def test_no_call_and_no_item_node_set_is_none():
    assert _union_node_sets(None, None) is None


def test_call_only_is_returned_unchanged():
    call_node_set = ["a", "b"]
    assert _union_node_sets(call_node_set, None) is call_node_set


def test_call_only_duplicates_are_preserved_not_deduped():
    # No item-level node_set: behavior must match pre-existing code exactly,
    # including a caller's own duplicate entries.
    call_node_set = ["a", "a", "b"]
    assert _union_node_sets(call_node_set, None) == ["a", "a", "b"]


def test_item_only_is_returned_as_is():
    assert _union_node_sets(None, ["x"]) == ["x"]


def test_call_and_item_union_call_first_deduped():
    assert _union_node_sets(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_empty_item_list_behaves_like_none():
    call_node_set = ["a", "a"]
    assert _union_node_sets(call_node_set, []) is call_node_set


def test_the_union_reaches_document_belongs_to_set():
    """End-to-end check of the path ingest_data actually writes: the unioned
    list goes into external_metadata["node_set"], and classify_documents
    reads exactly that key to build the document's NodeSet membership."""
    effective_node_set = _union_node_sets(["call:x"], ["notion:ws:root"])

    document = TextDocument(
        id=uuid4(),
        title="page.md",
        name="page",
        raw_data_location="/tmp/page.md",
        mime_type="text/markdown",
        external_metadata=json.dumps({"node_set": effective_node_set}),
    )
    update_node_set(document)

    assert {node.name for node in document.belongs_to_set} == {"call:x", "notion:ws:root"}


def test_union_dedupes_on_the_node_set_id_key():
    assert _union_node_sets(["notion:A B"], ["notion:a_b", "notion:c"]) == [
        "notion:A B",
        "notion:c",
    ]


def test_a_string_node_set_is_one_name_not_characters():
    assert _union_node_sets("notion:x", ["notion:y"]) == ["notion:x", "notion:y"]
    assert _union_node_sets(None, "notion:x") == ["notion:x"]
