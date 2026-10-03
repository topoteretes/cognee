"""PATCH rule for a re-ingested row's metadata and node set (SDK-750).

A re-ingest of an existing document (a plain re-add, or update()'s full rebuild)
must treat a field the request does not carry as "keep", an explicit empty value
as "clear", and a value as "replace". Until SDK-750 external_metadata and node_set
were overwritten from the request alone, so a full rebuild wiped whatever the
caller did not resend.
"""

import json

from cognee.tasks.ingestion.ingest_data import merge_existing_row_metadata

STORED = {"source": "audit", "node_set": ["ns1"]}
STORED_NODE_SET = json.dumps(["ns1"])


def test_absent_keeps_stored_metadata_and_node_set():
    metadata, node_set = merge_existing_row_metadata(STORED, STORED_NODE_SET, {}, None, None)

    assert metadata == {"source": "audit", "node_set": ["ns1"]}
    assert node_set == ["ns1"]


def test_empty_dict_clears_metadata_but_keeps_node_set():
    metadata, node_set = merge_existing_row_metadata(STORED, STORED_NODE_SET, {}, {}, None)

    assert metadata == {"node_set": ["ns1"]}
    assert node_set == ["ns1"]


def test_empty_list_clears_node_set_but_keeps_metadata():
    metadata, node_set = merge_existing_row_metadata(STORED, STORED_NODE_SET, {}, None, [])

    assert metadata == {"source": "audit"}
    assert node_set is None


def test_values_replace():
    metadata, node_set = merge_existing_row_metadata(
        STORED, STORED_NODE_SET, {}, {"source": "crm"}, ["ns2"]
    )

    assert metadata == {"source": "crm", "node_set": ["ns2"]}
    assert node_set == ["ns2"]


def test_content_metadata_refreshes_on_top_of_kept_metadata():
    """Metadata derived from the new content describes the new content, so it wins
    over the stored copy even when the caller asked to keep everything."""
    stored = {"source": "audit", "origin": "old", "node_set": ["ns1"]}

    metadata, _ = merge_existing_row_metadata(
        stored, STORED_NODE_SET, {"origin": "new", "metadata": {"page": 3}}, None, None
    )

    assert metadata == {
        "source": "audit",
        "origin": "new",
        "metadata": {"page": 3},
        "node_set": ["ns1"],
    }


def test_row_without_stored_values_stays_empty_when_nothing_is_sent():
    metadata, node_set = merge_existing_row_metadata(None, None, {}, None, None)

    assert metadata == {}
    assert node_set is None


def test_stored_input_is_not_mutated():
    stored = dict(STORED)

    merge_existing_row_metadata(stored, STORED_NODE_SET, {"origin": "x"}, None, [])

    assert stored == STORED
