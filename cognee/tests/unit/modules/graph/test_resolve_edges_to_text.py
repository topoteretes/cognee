"""
Unit Tests: resolve_edges_to_text

Pins the contract that the bracket label is a compact relationship label
(not the natural-language edge_text) and that edge_text, when present, is
surfaced alongside the markup rather than inside it.
"""

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text


@pytest.mark.asyncio
@pytest.mark.parametrize("json_marks", [False, True])
async def test_conflict_labels_reorder_only_selected_connections(json_marks):
    marks = (
        {"conflict_marks_json": '[{"conflict_id":"f1","status":"superseded"}]'}
        if json_marks
        else {"conflict_marks": [{"conflict_id": "f1", "status": "superseded"}]}
    )
    old = _make_edge(
        "Acme", "Alice", {"relationship_type": "has_ceo", "effective_date": "2020-05-01", **marks}
    )
    current = _make_edge(
        "Acme",
        "Bob",
        {
            "relationship_type": "has_ceo",
            "effective_date": "2026-01-10",
            "conflict_marks": [{"conflict_id": "f1", "status": "current"}],
        },
    )
    selected = [old, current]
    text = await resolve_edges_to_text(selected)
    connections = text.split("Connections:\n", 1)[1]
    assert connections.splitlines() == [
        "Acme --[has_ceo]--> Bob [as of 2026-01-10]",
        "Acme --[has_ceo]--> Alice [superseded; as of 2020-05-01]",
    ]
    assert selected == [old, current]
    # The formatter may reorder a top-k selection but never drops its historical hit.
    text = await resolve_edges_to_text([old])
    assert "Alice [superseded; as of 2020-05-01]" in text
    assert "Bob" not in text and "## Fact conflicts" not in text


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
