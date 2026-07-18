"""Unit tests for the connector-agnostic DLT document-mode seam.

A dlt source opts into the "document" ingestion path (LLM entity extraction)
by setting ``DOCUMENT_SOURCE_ATTR`` on itself; resolve_dlt_sources then tags its
rows ``external_metadata["source"] = <tag>`` (NOT "dlt"), so ``is_dlt_sourced``
returns False and classify_documents routes them to TextDocument/cognify rather
than the deterministic DltRowDocument schema path. These tests exercise that seam
with plain objects — no connector, no database, no LLM.
"""

from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid5

from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    document_source_tag,
    is_dlt_sourced,
)
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item


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
    assert item.external_metadata["source"] == "notion"
    assert is_dlt_sourced(item.external_metadata) is False
    assert item.external_metadata["url"] == "https://example.com/p1"
    assert item.external_metadata["external_id"] == "p1"
    assert item.data_id == data_id
    # title becomes an H1 prefixed to the content body.
    assert item.data.startswith("# My Page")
    assert "body text" in item.data


def test_build_document_data_item_without_title_is_just_content():
    row = SimpleNamespace(
        row_data={"id": "x", "content": "plain body"},
        content_hash="h",
    )
    item = _build_document_data_item(row, uuid5(NAMESPACE_OID, "x"), "wiki")
    assert item.data == "plain body"
    assert item.external_metadata["source"] == "wiki"
    assert item.external_metadata["title"] is None
