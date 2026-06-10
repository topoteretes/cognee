"""Unit tests for the LLM-free reference (Evidence) helpers.

Covers ``format_chunk_references`` (sync, payload-driven) and
``build_graph_reference_context`` (async, GraphDBInterface-driven), including the
old-data graceful-degradation cases and the non-traversable-backend case.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.utils.references import (
    EVIDENCE_HEADER,
    build_graph_reference_context,
    format_chunk_references,
)


# ---------------------------------------------------------------------------
# format_chunk_references
# ---------------------------------------------------------------------------


def _payload(**overrides):
    base = {
        "document_name": "annual_report.pdf",
        "chunk_index": 4,  # 1-based display number -> 5
        "text": "Revenue grew 12 percent year over year.",
    }
    base.update(overrides)
    return base


def test_format_chunk_references_renders_one_based_number_from_chunk_index():
    """chunk_index + 1 is rendered as the display number."""
    result = format_chunk_references([_payload(chunk_index=4)])

    assert result.startswith(EVIDENCE_HEADER + "\n")
    assert "- chunk 5 of document annual_report.pdf:" in result
    assert "Revenue grew 12 percent" in result


def test_format_chunk_references_prefers_explicit_chunk_number():
    """An explicit chunk_number wins over chunk_index when both present."""
    result = format_chunk_references([_payload(chunk_number=9, chunk_index=4)])

    assert "- chunk 9 of document annual_report.pdf:" in result
    assert "chunk 5" not in result


def test_format_chunk_references_reads_scored_result_like_objects():
    """ScoredResult-like objects expose .payload and .id; they are read correctly."""

    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    objs = [FakeScored(_payload(), "id-1")]
    result = format_chunk_references(objs)

    assert "- chunk 5 of document annual_report.pdf:" in result


def test_format_chunk_references_empty_when_document_name_missing():
    """Old-data case: missing document_name -> entry skipped -> empty string."""
    payload = _payload()
    del payload["document_name"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_when_document_name_null():
    """Null document_name is unusable -> empty string."""
    assert format_chunk_references([_payload(document_name=None)]) == ""


def test_format_chunk_references_empty_when_no_chunk_number():
    """Missing both chunk_number and chunk_index -> empty string."""
    payload = _payload()
    del payload["chunk_index"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_when_text_missing():
    """Missing text -> no usable snippet -> empty string."""
    payload = _payload()
    del payload["text"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_for_empty_input():
    assert format_chunk_references([]) == ""
    assert format_chunk_references(None) == ""


def test_format_chunk_references_dedups_by_id():
    """Two payloads with the same object id collapse into one bullet."""

    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    objs = [
        FakeScored(_payload(text="first"), "same-id"),
        FakeScored(_payload(text="second"), "same-id"),
    ]
    result = format_chunk_references(objs)

    assert result.count("- chunk 5 of document annual_report.pdf:") == 1


def test_format_chunk_references_caps_and_clamps_limit():
    """Limit is clamped into the 3-5 range."""
    payloads = [
        _payload(document_name=f"doc_{i}.pdf", chunk_index=i, text=f"text {i}", id=str(i))
        for i in range(10)
    ]
    # limit below the floor is clamped up to 3
    low = format_chunk_references(payloads, limit=1)
    assert low.count("- chunk ") == 3
    # limit above the ceiling is clamped down to 5
    high = format_chunk_references(payloads, limit=99)
    assert high.count("- chunk ") == 5


def test_format_chunk_references_snippet_truncated():
    """Long text is truncated with an ellipsis."""
    long_text = "word " * 100
    result = format_chunk_references([_payload(text=long_text)])
    # The bullet line contains a truncation ellipsis.
    assert "…" in result


# ---------------------------------------------------------------------------
# build_graph_reference_context
# ---------------------------------------------------------------------------


def _entity_node():
    return {"id": "entity-1", "name": "Acme Corp", "type": "Entity"}


def _chunk_node(with_flat_name=True):
    node = {"id": "chunk-1", "name": "chunk-1", "type": "DocumentChunk", "chunk_index": 2}
    if with_flat_name:
        node["document_name"] = "annual_report.pdf"
    return node


def _document_node():
    return {"id": "doc-1", "name": "annual_report.pdf", "type": "Document"}


@pytest.mark.asyncio
async def test_build_graph_reference_context_uses_flat_document_name():
    """Entity -> contains -> chunk with a flat document_name resolves directly."""
    engine = AsyncMock()
    engine.get_connections.return_value = [
        (_entity_node(), {"relationship_name": "contains"}, _chunk_node(with_flat_name=True)),
    ]

    result = await build_graph_reference_context(["entity-1"], engine)

    assert result.startswith(EVIDENCE_HEADER + "\n")
    assert "- Entity Acme Corp appears in chunk 3 of document annual_report.pdf" in result
    # Only the single get_connections call was needed (flat name short-circuits).
    engine.get_connections.assert_awaited_once_with("entity-1")


@pytest.mark.asyncio
async def test_build_graph_reference_context_falls_back_to_is_part_of():
    """Without a flat name, it walks is_part_of -> Document for the name."""
    engine = AsyncMock()

    async def get_connections(node_id):
        if node_id == "entity-1":
            return [
                (
                    _entity_node(),
                    {"relationship_name": "contains"},
                    _chunk_node(with_flat_name=False),
                )
            ]
        if node_id == "chunk-1":
            return [
                (
                    _chunk_node(with_flat_name=False),
                    {"relationship_name": "is_part_of"},
                    _document_node(),
                )
            ]
        return []

    engine.get_connections.side_effect = get_connections

    result = await build_graph_reference_context(["entity-1"], engine)

    assert "- Entity Acme Corp appears in chunk 3 of document annual_report.pdf" in result


@pytest.mark.asyncio
async def test_build_graph_reference_context_returns_empty_on_not_implemented():
    """Non-traversable backend (Postgres-graph) raises NotImplementedError -> empty, no raise."""
    engine = AsyncMock()
    engine.get_connections.side_effect = NotImplementedError("graph traversal not supported")

    result = await build_graph_reference_context(["entity-1"], engine)

    assert result == ""


@pytest.mark.asyncio
async def test_build_graph_reference_context_returns_empty_on_attribute_error():
    """Engine missing get_connections -> AttributeError -> empty, no raise."""
    engine = AsyncMock()
    engine.get_connections.side_effect = AttributeError("no get_connections")

    result = await build_graph_reference_context(["entity-1"], engine)

    assert result == ""


@pytest.mark.asyncio
async def test_build_graph_reference_context_empty_for_no_node_ids_or_engine():
    assert await build_graph_reference_context([], AsyncMock()) == ""
    assert await build_graph_reference_context(["entity-1"], None) == ""


@pytest.mark.asyncio
async def test_build_graph_reference_context_empty_when_no_chunk_connections():
    """Entity with no contains->DocumentChunk connection produces no evidence."""
    engine = AsyncMock()
    engine.get_connections.return_value = [
        (_entity_node(), {"relationship_name": "related_to"}, _entity_node()),
    ]

    result = await build_graph_reference_context(["entity-1"], engine)

    assert result == ""
