"""Unit test for the Entity/EntityType node-ID namespacing migration.

Drives the migration against an in-memory fake graph engine that mimics the
``GraphDBInterface`` round-trip (``get_graph_data`` returns ``(id, props)`` with
``id`` split out; ``add_nodes`` consumes ``model_dump()``), so no real database
or LLM is required.
"""

import asyncio

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils.generate_node_name import generate_node_name
from cognee.modules.migrations.graph.namespace_entity_type_node_ids import migrate


class _FakeGraph:
    """Minimal in-memory graph mirroring the adapter's data shapes."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}  # id -> properties (without id)
        self.edges: list[tuple] = []  # (source, target, rel, props)

    def add_old_node(self, raw_name: str, node_type: str) -> str:
        # Reproduce the released BARE scheme: id = generate_node_id(name).
        name = generate_node_name(raw_name)
        node_id = str(generate_node_id(raw_name))
        self.nodes[node_id] = {"name": name, "type": node_type, "description": name}
        return node_id

    def add_interim_node(self, raw_name: str, node_type: str) -> str:
        # Reproduce the interim (never-released) "entity:"/"type:" scheme.
        prefix = "type" if node_type == "EntityType" else "entity"
        name = generate_node_name(raw_name)
        node_id = str(generate_node_id(f"{prefix}:{raw_name}"))
        self.nodes[node_id] = {"name": name, "type": node_type, "description": name}
        return node_id

    def add_raw_node(self, node_id: str, props: dict) -> str:
        self.nodes[node_id] = props
        return node_id

    async def get_graph_data(self):
        return [(nid, dict(props)) for nid, props in self.nodes.items()], list(self.edges)

    async def add_nodes(self, nodes):
        for node in nodes:
            data = node.model_dump()
            nid = data.pop("id")
            self.nodes[nid] = data

    async def add_edges(self, edges):
        self.edges.extend(edges)

    async def delete_nodes(self, node_ids):
        ids = set(node_ids)
        for nid in ids:
            self.nodes.pop(nid, None)
        self.edges = [e for e in self.edges if e[0] not in ids and e[1] not in ids]


def test_migration_remaps_entity_and_type_ids_and_edges():
    graph = _FakeGraph()

    # Old-scheme graph: EntityType "Person", Entity "Alice", and a DocumentChunk
    # that must be left untouched. Note the Entity/EntityType name collision
    # that #2515 fixes is exercised by "Person"/"person" sharing nothing here;
    # we focus on correct remapping + edge rewiring.
    type_old = graph.add_old_node("Person", "EntityType")
    entity_old = graph.add_old_node("Alice", "Entity")
    chunk_id = "chunk-1"
    graph.nodes[chunk_id] = {"name": "chunk", "type": "DocumentChunk", "description": "c"}

    graph.edges = [
        (entity_old, type_old, "is_a", {}),
        (chunk_id, entity_old, "contains", {"w": 1}),
    ]

    asyncio.run(migrate(graph))

    type_new = str(EntityType.id_for("person"))
    entity_new = str(Entity.id_for("alice"))

    # Old IDs gone, new namespaced IDs present, chunk untouched.
    assert type_old not in graph.nodes and entity_old not in graph.nodes
    assert type_new in graph.nodes and entity_new in graph.nodes
    assert chunk_id in graph.nodes
    # Properties preserved.
    assert graph.nodes[entity_new]["name"] == "alice"
    assert graph.nodes[type_new]["type"] == "EntityType"

    # Edges rewired onto the new IDs (old edges dropped with the old nodes).
    assert (entity_new, type_new, "is_a", {}) in graph.edges
    assert (chunk_id, entity_new, "contains", {"w": 1}) in graph.edges
    assert all(entity_old not in (e[0], e[1]) and type_old not in (e[0], e[1]) for e in graph.edges)


def test_migration_is_idempotent_on_new_scheme():
    """Running against an already-migrated graph is a no-op."""
    graph = _FakeGraph()
    type_new = str(EntityType.id_for("person"))
    entity_new = str(Entity.id_for("alice"))
    graph.nodes[type_new] = {"name": "person", "type": "EntityType", "description": "p"}
    graph.nodes[entity_new] = {"name": "alice", "type": "Entity", "description": "a"}
    before = ({k: dict(v) for k, v in graph.nodes.items()}, list(graph.edges))

    asyncio.run(migrate(graph))

    assert ({k: dict(v) for k, v in graph.nodes.items()}, list(graph.edges)) == before


def test_migration_remaps_interim_prefixed_scheme():
    """The never-released interim 'entity:'/'type:' scheme is also recognized."""
    graph = _FakeGraph()
    t_old = graph.add_interim_node("Person", "EntityType")
    e_old = graph.add_interim_node("Alice", "Entity")

    asyncio.run(migrate(graph))

    assert t_old not in graph.nodes and e_old not in graph.nodes
    assert str(EntityType.id_for("person")) in graph.nodes
    assert str(Entity.id_for("alice")) in graph.nodes


def test_migration_skips_unrecognized_ids():
    """A node whose id is not a recognized hash of its name is left untouched."""
    graph = _FakeGraph()
    # id is unrelated to the name (e.g. a random uuid or a different-field hash).
    weird_id = "00000000-0000-0000-0000-0000000000ff"
    graph.add_raw_node(weird_id, {"name": "alice", "type": "Entity", "description": "a"})
    before = {k: dict(v) for k, v in graph.nodes.items()}

    asyncio.run(migrate(graph))

    assert graph.nodes == before  # unchanged
    assert weird_id in graph.nodes


def test_make_node_satisfies_all_adapter_contracts():
    """The carrier must work for model_dump (Ladybug/Postgres) AND dict()+class
    name (Neo4j label) — the bug that broke the migration on Neo4j."""
    from cognee.modules.migrations.graph.namespace_entity_type_node_ids import _make_node

    props = {"id": "abc", "name": "alice", "type": "Entity", "description": "a"}
    node = _make_node(props)
    assert node.model_dump() == props  # Ladybug / Postgres path
    assert dict(node) == props  # Neo4j path: object must be iterable
    assert type(node).__name__ == "Entity"  # Neo4j uses this as the node label

    type_node = _make_node({"id": "x", "name": "person", "type": "EntityType"})
    assert type(type_node).__name__ == "EntityType"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
