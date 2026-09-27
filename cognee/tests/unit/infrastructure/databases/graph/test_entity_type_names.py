"""``get_entity_type_names``: an entity's EntityType through its ``is_a`` edge.

A bounded graph read can admit an entity without its EntityType node, and the
graph view still has to label it (SDK-794). The lookup reads one hop from the
entities only; starting from a type node would pull in every entity of it.
"""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

NODES = {
    "alice": {"type": "Entity", "name": "Alice"},
    "bob": {"type": "Entity", "name": "Bob"},
    "carol": {"type": "Entity", "name": "Carol"},
    "person": {"type": "EntityType", "name": "Person"},
}
EDGES = [
    ("alice", "person", "is_a"),
    ("bob", "person", "is_a"),
    ("alice", "bob", "knows"),
]


class _NeighborhoodOnly:
    """No native lookup: exercises the inherited default, invoked unbound."""

    def __init__(self):
        self.calls = []

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        self.calls.append((list(node_ids), depth, edge_types))
        members = set(node_ids)
        for source, target, relation in EDGES:
            if (not edge_types or relation in edge_types) and (
                source in node_ids or target in node_ids
            ):
                members |= {source, target}
        # Like the real adapters: every edge among the returned nodes.
        edges = [(s, t, r, {}) for s, t, r in EDGES if s in members and t in members]
        return [(n, dict(NODES[n])) for n in members], edges


@pytest.mark.asyncio
async def test_default_maps_entities_to_their_type_and_skips_other_edges():
    adapter = _NeighborhoodOnly()
    names = await GraphDBInterface.get_entity_type_names(adapter, ["alice", "carol"])
    assert names == {"alice": "Person"}
    assert adapter.calls == [(["alice", "carol"], 1, ["is_a"])]


@pytest.mark.asyncio
async def test_default_reads_nothing_for_no_entities():
    adapter = _NeighborhoodOnly()
    assert await GraphDBInterface.get_entity_type_names(adapter, []) == {}
    assert adapter.calls == []


def test_the_default_is_inherited_not_abstract():
    assert "get_entity_type_names" not in GraphDBInterface.__abstractmethods__


async def _fill(adapter):
    await adapter.initialize()
    for node_id, properties in NODES.items():
        await adapter.add_node(
            SimpleNamespace(model_dump=lambda node_id=node_id, p=properties: {"id": node_id, **p})
        )
    for source, target, relation in EDGES:
        await adapter.add_edge(source, target, relation)


@pytest.mark.asyncio
async def test_turso_native_lookup(tmp_path):
    from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

    adapter = TursoAdapter(f"sqlite+aiosqlite:///{tmp_path / 'types.db'}")
    try:
        await _fill(adapter)
        assert await adapter.get_entity_type_names(["alice", "bob", "carol"]) == {
            "alice": "Person",
            "bob": "Person",
        }
        # Never outward from a type node.
        assert await adapter.get_entity_type_names(["person"]) == {}
    finally:
        await adapter.engine.dispose()


@pytest.mark.asyncio
async def test_ladybug_native_lookup(tmp_path):
    pytest.importorskip("ladybug")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter(db_path=str(tmp_path / "types.lbug"))
    for node_id, properties in NODES.items():
        await adapter.query(
            "CREATE (:Node {id: $id, name: $name, type: $type, properties: '{}'})",
            {"id": node_id, **properties},
        )
    for source, target, relation in EDGES:
        await adapter.query(
            "MATCH (a:Node {id: $s}), (b:Node {id: $t}) "
            "CREATE (a)-[:EDGE {relationship_name: $r, properties: '{}'}]->(b)",
            {"s": source, "t": target, "r": relation},
        )
    assert await adapter.get_entity_type_names(["alice", "bob", "carol"]) == {
        "alice": "Person",
        "bob": "Person",
    }
    assert await adapter.get_entity_type_names(["person"]) == {}
