"""Tests for ingest_data's call-level + item-level node_set union.

ingest_data writes the union of the call's node_set param and a DataItem's
own node_set to external_metadata["node_set"] and Data.node_set. With no
item-level node_set the union must be a no-op: byte-identical to the
pre-existing call-level-only behavior, duplicates and all.
"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.modules.engine.models.node_set import InvalidNodeSetError
from cognee.tasks.documents.classify_documents import update_node_set
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.ingest_data import _union_node_sets, ingest_data

USER = SimpleNamespace(id=uuid4(), tenant_id=None)


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


# A malformed node_set used to be ignored, or tagged depending on whether an
# item had its own node_set. ingest_data now rejects it before storing anything.
_MALFORMED = ["team", ["a", 5], ["x", ""], ["x", "  "], ["x\ud800"], {"a": 1}]


@pytest.mark.asyncio
@pytest.mark.parametrize("node_set", _MALFORMED)
async def test_ingest_data_rejects_a_malformed_call_node_set(node_set):
    with pytest.raises(InvalidNodeSetError):
        await ingest_data("text", "ds", USER, node_set=node_set)


@pytest.mark.asyncio
@pytest.mark.parametrize("node_set", _MALFORMED)
async def test_ingest_data_rejects_a_malformed_item_node_set(node_set):
    with pytest.raises(InvalidNodeSetError):
        await ingest_data([DataItem(data="text", node_set=node_set)], "ds", USER)


@pytest.mark.asyncio
@pytest.mark.parametrize("node_set", _MALFORMED)
async def test_ingest_data_rejects_a_malformed_node_set_in_external_metadata(node_set):
    item = DataItem(data="text", external_metadata={"node_set": node_set})
    with pytest.raises(InvalidNodeSetError):
        await ingest_data([item], "ds", USER)


@pytest.mark.parametrize("node_set", _MALFORMED)
def test_a_stored_malformed_node_set_raises_instead_of_leaving_the_document_untagged(
    node_set,
):
    document = TextDocument(
        id=uuid4(),
        title="doc.txt",
        name="doc",
        raw_data_location="/tmp/doc.txt",
        external_metadata=json.dumps({"node_set": node_set}),
        mime_type="text/plain",
    )
    with pytest.raises(InvalidNodeSetError):
        update_node_set(document)
