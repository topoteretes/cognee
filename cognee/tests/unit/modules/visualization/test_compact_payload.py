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
    compact_chunk,
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
        accumulator.add(*compact_chunk(chunk_nodes, chunk_edges))
    summary = accumulator.summary()

    # A late entity_type correction, where present, is the type preprocess gives.
    by_id = {node["id"]: node for node in pre.nodes}
    for node_id, entry in summary["nodes"].items():
        if "entity_type" in entry:
            assert entry.pop("entity_type") == by_id[node_id]["entity_type"]
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


def _json_and_stream_types(nodes_data, edges_data):
    from cognee.modules.visualization.preprocessor import compact_chunk, preprocess

    json_types = {n["id"]: n["entity_type"] for n in preprocess((nodes_data, edges_data)).nodes}
    stream_nodes, _ = compact_chunk(nodes_data, edges_data)
    return json_types, {node["id"]: node["entity_type"] for node in stream_nodes}


def test_json_and_stream_agree_on_entity_type():
    """SDK-794: an entity whose type node is not in the read keeps its type in both."""
    from cognee.modules.visualization.preprocessor import SEMANTIC_TYPE_KEY

    nodes_data = [
        ("alice", {"type": "Entity", "name": "Alice", SEMANTIC_TYPE_KEY: "Person"}),
        ("bob", {"type": "Entity", "name": "Bob"}),
        ("chunk", {"type": "DocumentChunk", "text": "Alice knows Bob"}),
        ("untyped", {"name": "x"}),
    ]
    json_types, stream_types = _json_and_stream_types(nodes_data, [])
    assert json_types == stream_types
    assert json_types == {
        "alice": "Person",
        "bob": "Entity",
        "chunk": "DocumentChunk",
        "untyped": "Node",
    }


def test_json_and_stream_agree_when_only_the_is_a_link_types_an_entity():
    """The store lookup did not type alice (it failed, or a full read), her link does."""
    nodes_data = [
        ("alice", {"type": "Entity", "name": "Alice"}),
        ("person", {"type": "EntityType", "name": "Person"}),
    ]
    edges_data = [("alice", "person", "is_a", {})]
    json_types, stream_types = _json_and_stream_types(nodes_data, edges_data)
    assert json_types == stream_types == {"alice": "Person", "person": "EntityType"}


def test_the_internal_type_key_does_not_reach_the_payload():
    from cognee.modules.visualization.preprocessor import SEMANTIC_TYPE_KEY, preprocess

    pre = preprocess(([("alice", {"type": "Entity", "name": "A", SEMANTIC_TYPE_KEY: "P"})], []))
    assert SEMANTIC_TYPE_KEY not in pre.nodes[0]
    assert pre.nodes[0]["entity_type"] == "P"


def test_types_resolved_from_the_store_rank_and_roll_up_like_linked_ones():
    """Type cards from the store lookup count as entity types and hit the cap."""
    from cognee.modules.visualization.preprocessor import (
        OTHER_ENTITY_TYPES_LABEL,
        SCHEMA_MAX_ENTITY_TYPES,
        SEMANTIC_TYPE_KEY,
        extract_type_schema_graph_data,
    )

    count = SCHEMA_MAX_ENTITY_TYPES + 5
    entities = [{"id": f"e{i}", "type": "Entity", "name": f"E{i}"} for i in range(count)]
    type_nodes = [{"id": f"t{i}", "type": "EntityType", "name": f"Type{i}"} for i in range(count)]
    is_a = [{"source": f"e{i}", "target": f"t{i}", "relation": "is_a"} for i in range(count)]
    resolved = [{**node, SEMANTIC_TYPE_KEY: f"Type{i}"} for i, node in enumerate(entities)]

    def entity_cards(nodes, links):
        schema = extract_type_schema_graph_data(nodes, links)
        return {
            n["name"]
            for n in schema["nodes"]
            if n["name"].startswith("Type")
            and " to " not in n["name"]
            or n["name"] == OTHER_ENTITY_TYPES_LABEL
        }

    linked = entity_cards(entities + type_nodes, is_a)
    from_store = entity_cards(resolved, [])
    assert OTHER_ENTITY_TYPES_LABEL in from_store
    assert len(from_store) == SCHEMA_MAX_ENTITY_TYPES
    assert from_store == linked
