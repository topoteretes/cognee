"""The streamed graph must render the same as the JSON one.

The streamed /visualize/json computes each chunk from that chunk alone and the
graph-wide fields from a small accumulator, never from the whole graph with
its properties. What this pins is that nothing drifts from preprocess(): a
chunk node has the name, stage and placeholder flag preprocess() gives it, a
link has its edge class, and the summary equals what preprocess() assigns the
same graph in one piece.
"""

import random

import pytest

from cognee.infrastructure.databases.graph.bounded_neighborhood import chunk_members
from cognee.modules.visualization.preprocessor import (
    _NAME_FALLBACK_KEYS,
    COMPACT_PROPERTY_KEYS,
    CompactGraphAccumulator,
    compact_link,
    compact_node,
    preprocess,
)


def _graph(seed: int = 7, count: int = 120):
    """Nodes of every kind the naming rules treat differently, and random edges."""
    rng = random.Random(seed)
    kinds = [
        {"type": "Entity", "name": "Neon"},
        {"type": "DocumentChunk", "text": "  Neon was  chosen\nfor its latency. " * 20},
        {"type": "TextSummary", "summary": "Why neon won."},
        {"type": "Entity", "name": "3f2b8c1e-8a3f-4a51-9a6e-0f7d2c4b9e11"},
        {"type": "EntityType", "name": "Database"},
        {"type": "TextDocument", "name": "adr-12.md", "source_node_set": "docs"},
        {"type": "Entity", "name": "Rust", "belongs_to_set": ["session_learnings"]},
        {"type": "Mystery"},
    ]
    nodes = []
    for index in range(count):
        properties = dict(rng.choice(kinds))
        properties["noise"] = "x" * rng.randint(0, 50)
        properties["source_node_set"] = rng.choice([None, "docs", "code", "docs,code"])
        nodes.append((f"n{index}", properties))
    relations = ["contains", "is_a", "made_from", "mentions", "related_to"]
    edges = {
        (f"n{rng.randrange(count)}", f"n{rng.randrange(count)}", rng.choice(relations))
        for _ in range(count * 3)
    }
    return nodes, [(source, target, relation, {}) for source, target, relation in edges]


def test_the_projection_keeps_every_name_fallback():
    """Chunk nodes are named from their text; projecting it away renames them all."""
    assert set(_NAME_FALLBACK_KEYS) <= set(COMPACT_PROPERTY_KEYS)


def test_compact_nodes_match_preprocess():
    nodes, edges = _graph()
    expected = {node["id"]: node for node in preprocess((nodes, edges)).nodes}

    for node_id, properties in nodes:
        node = compact_node(node_id, properties)
        full = expected[node_id]
        assert node["name"] == full["name"]
        assert node["stage"] == full["stage"]
        assert node["is_unnamed"] == full["is_unnamed"]
        assert node["type"] == full["type"]


def test_projected_properties_are_enough_for_the_compact_node():
    nodes, _ = _graph()
    for node_id, properties in nodes:
        projected = {key: properties[key] for key in COMPACT_PROPERTY_KEYS if key in properties}
        projected["type"] = properties.get("type")
        assert compact_node(node_id, projected) == compact_node(node_id, properties)


def test_compact_links_match_preprocess():
    nodes, edges = _graph()
    expected = {
        (link["source"], link["target"], link["relation"]): link["edge_class"]
        for link in preprocess((nodes, edges)).links
    }

    for edge in edges:
        link = compact_link(edge)
        assert link["edge_class"] == expected[(link["source"], link["target"], link["relation"])]


@pytest.mark.parametrize("chunk_size", [1, 7, 50, 1000])
def test_the_summary_equals_preprocess_in_one_piece(chunk_size):
    nodes, edges = _graph()
    pre = preprocess((nodes, edges))

    accumulator = CompactGraphAccumulator()
    for chunk_nodes, chunk_edges in chunk_members(nodes, edges, chunk_size):
        accumulator.add(
            [compact_node(node_id, properties) for node_id, properties in chunk_nodes],
            [compact_link(edge) for edge in chunk_edges],
        )
    summary = accumulator.summary()

    assert summary["nodes"] == {
        node["id"]: {"importance": node["importance"], "label_priority": node["label_priority"]}
        for node in pre.nodes
    }
    assert summary["color_maps"]["node_set"] == pre.color_maps["node_set"]
    assert accumulator.node_count == len(pre.nodes)
    assert accumulator.link_count == len(pre.links)


def test_the_summary_of_an_empty_graph():
    summary = CompactGraphAccumulator().summary()

    assert summary == {"nodes": {}, "color_maps": {"node_set": {}}}
