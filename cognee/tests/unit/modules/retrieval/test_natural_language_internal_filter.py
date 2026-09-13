"""Internal nodes (is_internal marker) must never leak through natural-language search."""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.natural_language_retriever import (
    NaturalLanguageRetriever,
    _contains_internal_node,
    _is_internal_schema_row,
)


class TestContainsInternalNode:
    def test_flags_deserialized_internal_node_dict(self):
        assert _contains_internal_node({"is_internal": True, "preference_text": "terse"})

    def test_flags_internal_node_nested_in_row(self):
        row = {"n": {"id": "pref-1", "is_internal": True}}
        assert _contains_internal_node(row)

    def test_flags_internal_node_in_list_result(self):
        assert _contains_internal_node([{"name": "Alice"}, {"is_internal": True}])

    def test_flags_ladybug_serialized_properties_column(self):
        row = {"n": {"id": "pref-1", "properties": '{"is_internal": true, "user_id": "u1"}'}}
        assert _contains_internal_node(row)

    def test_passes_regular_rows(self):
        assert not _contains_internal_node({"n": {"id": "e-1", "name": "Alice"}})
        assert not _contains_internal_node([("node_count", 10)])
        assert not _contains_internal_node(None)

    def test_passes_plain_text_mentioning_the_marker(self):
        # Document text talking about "is_internal" is not an internal node.
        assert not _contains_internal_node({"text": "the is_internal flag hides nodes"})


class TestIsInternalSchemaRow:
    def test_flags_schema_row_listing_the_marker_key(self):
        row = {"NodeLabels": ["Node"], "Properties": ["name", "is_internal", "text"]}
        assert _is_internal_schema_row(row)

    def test_passes_schema_row_without_the_marker(self):
        row = {"NodeLabels": ["Node"], "Properties": ["name", "text"]}
        assert not _is_internal_schema_row(row)

    def test_passes_non_mapping_rows(self):
        assert not _is_internal_schema_row("is_internal-ish string")
        assert not _is_internal_schema_row(None)


@pytest.mark.asyncio
async def test_execute_cypher_query_filters_internal_rows():
    """Rows containing internal nodes are dropped from the raw Cypher result."""
    node_schema = [{"NodeLabels": ["Node"], "Properties": ["name"]}]
    edge_schema = [{"key": "relationship_name"}]
    raw_result = [
        {"n": {"id": "e-1", "name": "Alice"}},
        {"n": {"id": "pref-1", "is_internal": True, "preference_text": "terse"}},
    ]
    mock_engine = AsyncMock()
    mock_engine.query = AsyncMock(side_effect=[node_schema, edge_schema, raw_result])

    retriever = NaturalLanguageRetriever(max_attempts=1)
    with patch.object(
        NaturalLanguageRetriever,
        "_generate_cypher_query",
        AsyncMock(return_value="MATCH (n) RETURN n"),
    ):
        context = await retriever._execute_cypher_query("who is alice", mock_engine)

    assert context == [{"n": {"id": "e-1", "name": "Alice"}}]


@pytest.mark.asyncio
async def test_get_graph_schema_drops_internal_node_rows():
    """Schema rows describing internal nodes stay out of what the LLM can see."""
    node_schema = [
        {"NodeLabels": ["Node"], "Properties": ["name", "text"]},
        {"NodeLabels": ["Node"], "Properties": ["is_internal", "preference_text", "user_id"]},
    ]
    edge_schema = [{"key": "relationship_name"}]
    mock_engine = AsyncMock()
    mock_engine.query = AsyncMock(side_effect=[node_schema, edge_schema])

    retriever = NaturalLanguageRetriever()
    node_schemas, edge_schemas = await retriever._get_graph_schema(mock_engine)

    assert node_schemas == [{"NodeLabels": ["Node"], "Properties": ["name", "text"]}]
    assert edge_schemas == edge_schema
