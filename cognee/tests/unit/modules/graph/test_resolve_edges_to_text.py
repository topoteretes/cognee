"""
Unit Tests: resolve_edges_to_text

Pins the contract that the bracket label is a compact relationship label
(not the natural-language edge_text) and that edge_text, when present, is
surfaced alongside the markup rather than inside it.
"""

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text


def _make_edge(source_name: str, target_name: str, attributes: dict) -> Edge:
    source = Node(node_id=source_name, attributes={"name": source_name})
    target = Node(node_id=target_name, attributes={"name": target_name})
    return Edge(source, target, attributes=attributes)


@pytest.mark.asyncio
async def test_bracket_label_uses_relationship_type_not_edge_text():
    edge = _make_edge(
        "Alice",
        "Acme",
        {
            "relationship_type": "works_for",
            "edge_text": "Alice works at Acme as a platform engineer.",
        },
    )

    output = await resolve_edges_to_text([edge])

    assert "Alice --[works_for]--> Acme" in output


@pytest.mark.asyncio
async def test_edge_text_appears_as_suffix_when_different_from_label():
    description = "Alice works at Acme as a platform engineer."
    edge = _make_edge(
        "Alice",
        "Acme",
        {"relationship_type": "works_for", "edge_text": description},
    )

    output = await resolve_edges_to_text([edge])

    assert f"Alice --[works_for]--> Acme  ({description})" in output


@pytest.mark.asyncio
async def test_edge_text_suffix_omitted_when_equal_to_label():
    edge = _make_edge(
        "Alice",
        "Acme",
        {"relationship_type": "works_for", "edge_text": "works_for"},
    )

    output = await resolve_edges_to_text([edge])

    assert "Alice --[works_for]--> Acme" in output
    # No parenthetical suffix when edge_text equals the bracket label.
    assert "(works_for)" not in output


@pytest.mark.asyncio
async def test_falls_back_to_relationship_name_then_edge_text():
    edge_with_name_only = _make_edge(
        "Alice",
        "Acme",
        {"relationship_name": "works_for"},
    )
    edge_with_text_only = _make_edge(
        "Bob",
        "Globex",
        {"edge_text": "Bob works at Globex."},
    )

    output_name = await resolve_edges_to_text([edge_with_name_only])
    output_text = await resolve_edges_to_text([edge_with_text_only])

    assert "Alice --[works_for]--> Acme" in output_name
    assert "Bob --[Bob works at Globex.]--> Globex" in output_text
