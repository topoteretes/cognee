"""Scoped GRAPH_COMPLETION searches must not render entity descriptions that
another dataset's extraction wrote (description is a single global property on
shared entity nodes, overwritten last-writer-wins)."""

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.graph_completion_retriever import (
    GraphCompletionRetriever,
    _suppress_shared_entity_descriptions,
)


def _node(node_id, **attributes):
    return Node(node_id, dict(attributes))


def _edge(node1, node2, relationship_name="related_to"):
    return Edge(node1, node2, {"relationship_name": relationship_name})


def test_shared_entity_description_removed():
    shared = _node("shared", name="psv", description="foreign text", belongs_to_set=["A", "B"])
    local = _node("local", name="vessel", description="local text", belongs_to_set=["A"])
    edges = [_edge(shared, local)]

    _suppress_shared_entity_descriptions(edges)

    assert "description" not in shared.attributes
    assert local.attributes["description"] == "local text"


def test_untagged_node_description_removed_fail_closed():
    untagged = _node("untagged", name="mystery", description="unknown origin")
    local = _node("local", name="vessel", description="local text", belongs_to_set=["A"])
    edges = [_edge(untagged, local)]

    _suppress_shared_entity_descriptions(edges)

    assert "description" not in untagged.attributes
    assert local.attributes["description"] == "local text"


def test_chunk_nodes_with_text_untouched():
    chunk = _node(
        "chunk",
        text="raw chunk text",
        description="chunk description",
        belongs_to_set=["A", "B"],
    )
    local = _node("local", name="vessel", belongs_to_set=["A"])
    edges = [_edge(chunk, local)]

    _suppress_shared_entity_descriptions(edges)

    assert chunk.attributes["description"] == "chunk description"


@pytest.mark.asyncio
async def test_resolver_suppresses_only_when_scoped():
    def build_edges():
        shared = _node("shared", name="psv", description="foreign text", belongs_to_set=["A", "B"])
        local = _node("local", name="vessel", description="local text", belongs_to_set=["A"])
        return [_edge(shared, local)]

    scoped = GraphCompletionRetriever(node_name=["A"])
    scoped_text = await scoped.resolve_edges_to_text(build_edges())
    assert "foreign text" not in scoped_text
    assert "local text" in scoped_text

    unscoped = GraphCompletionRetriever()
    unscoped_text = await unscoped.resolve_edges_to_text(build_edges())
    assert "foreign text" in unscoped_text
    assert "local text" in unscoped_text
