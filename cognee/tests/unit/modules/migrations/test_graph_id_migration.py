"""Unit tests for the Entity/EntityType node-ID namespacing migration.

Drives the migration's pure store-helpers against in-memory fakes (no real
database or LLM):

  * ``build_id_remap`` + ``_migrate_graph`` — graph rename round-trip.
  * ``_migrate_vector`` / ``_migrate_triplet_vector`` — vector point re-key.

The fakes mirror the REAL adapter contracts, not what the migration would like
them to be: vector payloads are ``IndexSchema``-shaped (``{id, text,
belongs_to_set}`` — what ``index_data_points`` actually stores; never a dump of
the Entity model), and re-inserts go through ``index_data_points``. The fake
deliberately does NOT expose ``create_data_points``, so any regression back to
model-reconstruction fails loudly here.

The relational ledger step (``_migrate_ledger``) needs a real relational engine
and is covered by the backwards-compatibility / cross-adapter integration test.
"""

import asyncio

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils.generate_node_name import generate_node_name
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
    _build_triplet_remap,
    _make_node,
    _migrate_graph,
    _migrate_triplet_vector,
    _migrate_vector,
    _rekey_native,
    build_id_remap,
)


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


class _Row:
    """Mimics the ScoredResult shape returned by ``vector_engine.retrieve``."""

    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class _FakeVector:
    """In-memory vector store mirroring the REAL adapter contracts.

    Stores ``IndexSchema``-shaped payloads (``{id, text, belongs_to_set}``) and
    writes through ``index_data_points`` exactly like LanceDB/PGVector/Chroma:
    the row is rebuilt from the data point's id + embeddable field, never from
    the incoming object's full dump. No ``create_data_points`` on purpose — the
    migration must not depend on it.
    """

    def __init__(self):
        self.points: dict[str, dict] = {}  # collection -> {id: payload}

    def seed_index_point(self, collection, point_id, text, belongs_to_set=None):
        """Seed a point exactly as ``index_data_points`` would have stored it."""
        self.points.setdefault(collection, {})[str(point_id)] = {
            "id": str(point_id),
            "text": text,
            "belongs_to_set": belongs_to_set or [],
        }

    async def retrieve(self, collection, ids):
        store = self.points.get(collection, {})
        wanted = {str(i) for i in ids}
        return [_Row(id=pid, payload=payload) for pid, payload in store.items() if pid in wanted]

    async def index_data_points(self, index_name, index_property_name, data_points):
        collection = f"{index_name}_{index_property_name}"
        store = self.points.setdefault(collection, {})
        for data_point in data_points:
            store[str(data_point.id)] = {
                "id": str(data_point.id),
                "text": getattr(data_point, data_point.metadata["index_fields"][0]),
                "belongs_to_set": data_point.belongs_to_set or [],
            }

    async def delete_data_points(self, collection, ids):
        store = self.points.get(collection, {})
        for point_id in ids:
            store.pop(str(point_id), None)


class _NoCollectionVector(_FakeVector):
    """Adapter whose ``retrieve`` raises on a missing collection (e.g. PGVector)."""

    async def retrieve(self, collection, ids):
        from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

        if collection not in self.points:
            raise CollectionNotFoundError(f"collection {collection} does not exist")
        return await super().retrieve(collection, ids)


class _BrokenVector(_FakeVector):
    """Adapter whose ``retrieve`` fails with a REAL backend error (timeout etc.)."""

    async def retrieve(self, collection, ids):
        raise RuntimeError("connection timed out")


async def _run_graph_migration(graph):
    """Helper mirroring migrate()'s graph step from the in-memory fake's data."""
    nodes, edges = await graph.get_graph_data()
    id_map = build_id_remap(nodes)
    properties_by_id = {nid: props for nid, props in nodes}
    await _migrate_graph(graph, id_map, properties_by_id, edges)
    return id_map


# ── graph store ──────────────────────────────────────────────────────────────


def test_migration_remaps_entity_and_type_ids_and_edges():
    graph = _FakeGraph()

    type_old = graph.add_old_node("Person", "EntityType")
    entity_old = graph.add_old_node("Alice", "Entity")
    chunk_id = "chunk-1"
    graph.nodes[chunk_id] = {"name": "chunk", "type": "DocumentChunk", "description": "c"}

    graph.edges = [
        (entity_old, type_old, "is_a", {}),
        (chunk_id, entity_old, "contains", {"w": 1}),
    ]

    asyncio.run(_run_graph_migration(graph))

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


def test_migration_rewrites_endpoint_ids_embedded_in_edge_properties():
    """cognify embeds source_node_id/target_node_id INSIDE edge properties and
    retrieval prefers them over real topology — they must move with the edge."""
    graph = _FakeGraph()
    type_old = graph.add_old_node("Person", "EntityType")
    entity_old = graph.add_old_node("Alice", "Entity")
    graph.edges = [
        (
            entity_old,
            type_old,
            "is_a",
            {
                "relationship_name": "is_a",
                "source_node_id": entity_old,
                "target_node_id": type_old,
            },
        ),
    ]

    asyncio.run(_run_graph_migration(graph))

    type_new = str(EntityType.id_for("person"))
    entity_new = str(Entity.id_for("alice"))
    (source, target, rel, props) = graph.edges[0]
    assert (source, target, rel) == (entity_new, type_new, "is_a")
    assert props["source_node_id"] == entity_new
    assert props["target_node_id"] == type_new


def test_migration_does_not_persist_synthetic_self_edges():
    """Ladybug fabricates (id, id, 'SELF') edges for an edgeless graph; the
    migration must not turn them into real stored relationships."""
    graph = _FakeGraph()
    entity_old = graph.add_old_node("Alice", "Entity")
    graph.edges = [
        (entity_old, entity_old, "SELF", {"relationship_name": "SELF"}),
    ]

    asyncio.run(_run_graph_migration(graph))

    entity_new = str(Entity.id_for("alice"))
    assert entity_new in graph.nodes
    assert graph.edges == []  # old SELF dropped with the node, no new one written


def test_migration_is_idempotent_on_new_scheme():
    """Running against an already-migrated graph is a no-op."""
    graph = _FakeGraph()
    type_new = str(EntityType.id_for("person"))
    entity_new = str(Entity.id_for("alice"))
    graph.nodes[type_new] = {"name": "person", "type": "EntityType", "description": "p"}
    graph.nodes[entity_new] = {"name": "alice", "type": "Entity", "description": "a"}
    before = ({k: dict(v) for k, v in graph.nodes.items()}, list(graph.edges))

    asyncio.run(_run_graph_migration(graph))

    assert ({k: dict(v) for k, v in graph.nodes.items()}, list(graph.edges)) == before


def test_migration_remaps_interim_prefixed_scheme():
    """The never-released interim 'entity:'/'type:' scheme is also recognized."""
    graph = _FakeGraph()
    t_old = graph.add_interim_node("Person", "EntityType")
    e_old = graph.add_interim_node("Alice", "Entity")

    asyncio.run(_run_graph_migration(graph))

    assert t_old not in graph.nodes and e_old not in graph.nodes
    assert str(EntityType.id_for("person")) in graph.nodes
    assert str(Entity.id_for("alice")) in graph.nodes


def test_migration_skips_unrecognized_ids():
    """A node whose id is not a recognized hash of its name is left untouched."""
    graph = _FakeGraph()
    weird_id = "00000000-0000-0000-0000-0000000000ff"
    graph.add_raw_node(weird_id, {"name": "alice", "type": "Entity", "description": "a"})
    before = {k: dict(v) for k, v in graph.nodes.items()}

    asyncio.run(_run_graph_migration(graph))

    assert graph.nodes == before  # unchanged
    assert weird_id in graph.nodes


def test_make_node_satisfies_all_adapter_contracts():
    """The carrier must work for model_dump (Ladybug/Postgres) AND dict()+class
    name (Neo4j label) — the bug that broke the migration on Neo4j."""
    props = {"id": "abc", "name": "alice", "type": "Entity", "description": "a"}
    node = _make_node(props)
    assert node.model_dump() == props  # Ladybug / Postgres path
    assert dict(node) == props  # Neo4j path: object must be iterable
    assert type(node).__name__ == "Entity"  # Neo4j uses this as the node label

    type_node = _make_node({"id": "x", "name": "person", "type": "EntityType"})
    assert type(type_node).__name__ == "EntityType"


# ── vector store: entity points ──────────────────────────────────────────────


def test_vector_rekey_moves_real_indexschema_point():
    """A point stored the way real adapters store it ({id, text} — NO
    name/description) is re-keyed; reconstructing the Entity model from the
    payload must never be attempted."""
    vector = _FakeVector()
    old_id = str(generate_node_id("Alice"))
    new_id = str(Entity.id_for("alice"))
    vector.seed_index_point("Entity_name", old_id, "alice", belongs_to_set=["set-1"])

    id_map = {old_id: new_id}
    properties_by_id = {old_id: {"name": "alice", "type": "Entity", "description": "a"}}

    failed = asyncio.run(_migrate_vector(vector, id_map, properties_by_id))

    assert failed == set()
    store = vector.points["Entity_name"]
    assert old_id not in store
    # Re-inserted through index_data_points: same IndexSchema shape, text and
    # belongs_to_set preserved, id moved.
    assert store[new_id] == {"id": new_id, "text": "alice", "belongs_to_set": ["set-1"]}


def test_vector_rekey_falls_back_to_graph_property_for_text():
    """A payload without ``text`` re-embeds from the graph node's own index
    property instead of failing."""
    vector = _FakeVector()
    old_id = str(generate_node_id("Bob"))
    new_id = str(Entity.id_for("bob"))
    vector.points.setdefault("Entity_name", {})[old_id] = {"id": old_id}  # degenerate payload

    id_map = {old_id: new_id}
    properties_by_id = {old_id: {"name": "bob", "type": "Entity", "description": "b"}}

    failed = asyncio.run(_migrate_vector(vector, id_map, properties_by_id))

    assert failed == set()
    assert vector.points["Entity_name"][new_id]["text"] == "bob"


def test_vector_rekey_fails_point_with_no_text_anywhere():
    """No payload text and no graph property -> reported as failed and left in
    place (the caller keeps it on the old scheme everywhere)."""
    vector = _FakeVector()
    old_id = str(generate_node_id("Carol"))
    new_id = str(Entity.id_for("carol"))
    vector.points.setdefault("Entity_name", {})[old_id] = {"id": old_id}

    id_map = {old_id: new_id}
    properties_by_id = {old_id: {"type": "Entity"}}  # no name on the graph node either

    failed = asyncio.run(_migrate_vector(vector, id_map, properties_by_id))

    assert failed == {old_id}
    assert old_id in vector.points["Entity_name"]
    assert new_id not in vector.points["Entity_name"]


def test_vector_rekey_is_noop_without_points():
    """Graph-only deployments (no vector point for the id) re-key nothing."""
    vector = _FakeVector()
    old_id = str(generate_node_id("Carol"))
    new_id = str(Entity.id_for("carol"))

    failed = asyncio.run(
        _migrate_vector(vector, {old_id: new_id}, {old_id: {"name": "carol", "type": "Entity"}})
    )
    assert failed == set()
    assert vector.points == {}


# ── vector store: triplet points ─────────────────────────────────────────────


def _triplet_id(source: str, rel: str, target: str) -> str:
    """Mirror add_data_points._create_triplets_from_graph's id derivation."""
    return str(generate_node_id(source + rel + target))


def test_triplet_remap_follows_remapped_edge_endpoints():
    entity_old = str(generate_node_id("Alice"))
    entity_new = str(Entity.id_for("alice"))
    id_map = {entity_old: entity_new}
    edges = [
        ("chunk-1", entity_old, "contains", {}),
        ("chunk-1", "chunk-2", "next_chunk", {}),  # untouched endpoints -> no remap
        (entity_old, entity_old, "SELF", {}),  # synthetic placeholder -> skipped
    ]

    triplet_map = _build_triplet_remap(edges, id_map)

    assert triplet_map == {
        _triplet_id("chunk-1", "contains", entity_old): _triplet_id(
            "chunk-1", "contains", entity_new
        )
    }


def test_triplet_points_are_rekeyed_with_text_preserved():
    vector = _FakeVector()
    entity_old = str(generate_node_id("Alice"))
    entity_new = str(Entity.id_for("alice"))
    old_tid = _triplet_id("chunk-1", "contains", entity_old)
    new_tid = _triplet_id("chunk-1", "contains", entity_new)
    vector.seed_index_point("Triplet_text", old_tid, "chunk -› contains-›alice")

    asyncio.run(_migrate_triplet_vector(vector, {old_tid: new_tid}))

    store = vector.points["Triplet_text"]
    assert old_tid not in store
    assert store[new_tid]["text"] == "chunk -› contains-›alice"


def test_triplet_migration_tolerates_missing_collection():
    """Triplet embedding is opt-in; adapters that raise on a missing collection
    must not abort the migration."""
    vector = _NoCollectionVector()
    asyncio.run(_migrate_triplet_vector(vector, {"old": "new"}))  # must not raise
    assert vector.points == {}


def test_triplet_migration_propagates_real_backend_errors():
    """ONLY missing-collection is tolerated. A real backend failure (timeout,
    auth) must abort the migration so the revision is NOT stamped and the next
    startup retries — swallowing it would strand Triplet points forever."""
    import pytest

    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(_migrate_triplet_vector(_BrokenVector(), {"old": "new"}))


# ── vector store: native (vector-preserving) re-key paths ───────────────────


class _FakeLanceQuery:
    def __init__(self, table):
        self._table = table
        self._where = None

    def where(self, clause):
        self._where = clause
        return self

    async def to_list(self):
        import re

        wanted = re.findall(r"'([^']+)'", self._where or "")
        return [dict(self._table.rows[i]) for i in wanted if i in self._table.rows]


class _FakeLanceTable:
    """Mimics the lancedb AsyncTable surface the native re-key uses."""

    def __init__(self, rows):
        self.rows = {row["id"]: row for row in rows}
        self.optimized = False

    def query(self):
        return _FakeLanceQuery(self)

    async def add(self, new_rows):
        for row in new_rows:
            self.rows[row["id"]] = row

    async def optimize(self):
        self.optimized = True


def _make_lance_adapter(table):
    """Adapter stub whose CLASS NAME drives the native-path dispatch."""
    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    class LanceDBAdapter:
        def __init__(self):
            self.table = table

        async def get_collection(self, collection):
            if self.table is None:
                raise CollectionNotFoundError(f"Collection '{collection}' not found!")
            return self.table

        async def delete_data_points(self, collection, ids):
            for point_id in ids:
                self.table.rows.pop(str(point_id), None)

    return LanceDBAdapter()


def test_native_lancedb_rekey_moves_vector_without_reembedding():
    old_id = str(generate_node_id("Alice"))
    new_id = str(Entity.id_for("alice"))
    vector = [0.1, 0.2, 0.3]
    table = _FakeLanceTable(
        [{"id": old_id, "vector": vector, "payload": {"id": old_id, "text": "alice"}}]
    )
    adapter = _make_lance_adapter(table)

    handled = asyncio.run(_rekey_native(adapter, "Entity_name", {old_id: new_id}))

    assert handled
    assert old_id not in table.rows
    moved = table.rows[new_id]
    assert moved["vector"] == vector  # the stored vector travelled, byte for byte
    assert moved["payload"] == {"id": new_id, "text": "alice"}
    assert table.optimized  # compacted so later merge_inserts see no deletion vectors


def test_native_lancedb_missing_collection_is_noop():
    adapter = _make_lance_adapter(None)
    assert asyncio.run(_rekey_native(adapter, "Triplet_text", {"old": "new"}))


def test_native_pgvector_rekey_updates_rows_in_place():
    """Drives the real SQL path against an in-memory SQLAlchemy table shaped
    like a PGVector collection (id PK, payload JSON, vector)."""
    from sqlalchemy import JSON, Column, MetaData, String, Table, select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    old_id = str(generate_node_id("Alice"))
    new_id = str(Entity.id_for("alice"))

    async def run():
        engine = create_async_engine("sqlite+aiosqlite://")
        metadata = MetaData()
        table = Table(
            "Entity_name",
            metadata,
            Column("id", String, primary_key=True),
            Column("payload", JSON),
            Column("vector", JSON),
        )
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(
                table.insert().values(
                    id=old_id, payload={"id": old_id, "text": "alice"}, vector=[0.1, 0.2]
                )
            )

        class PGVectorAdapter:  # class name drives the dispatch
            async def get_table(self, collection):
                return table

            def get_async_session(self):
                return AsyncSession(engine)

        handled = await _rekey_native(PGVectorAdapter(), "Entity_name", {old_id: new_id})

        async with AsyncSession(engine) as session:
            rows = (await session.execute(select(table))).all()
        await engine.dispose()
        return handled, rows

    handled, rows = asyncio.run(run())
    assert handled
    assert len(rows) == 1
    assert rows[0].id == new_id
    assert rows[0].payload == {"id": new_id, "text": "alice"}
    assert rows[0].vector == [0.1, 0.2]  # untouched — nothing re-embedded


def test_rekey_native_unknown_adapter_falls_back():
    """Backends without a native path report unhandled so the caller re-embeds."""
    assert not asyncio.run(_rekey_native(_FakeVector(), "Entity_name", {"old": "new"}))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))


# ── merge safety (pre-existing new-scheme rows must never duplicate) ─────────


def test_native_lancedb_rekey_merges_when_new_id_already_exists():
    """An SDK process may have written the new-scheme point before the
    migration ran. The native re-key must DROP the old row, not append a
    duplicate id (lance `add` has no uniqueness)."""
    old_id = str(generate_node_id("Alice"))
    new_id = str(Entity.id_for("alice"))
    table = _FakeLanceTable(
        [
            {"id": old_id, "vector": [0.1], "payload": {"id": old_id, "text": "alice"}},
            {"id": new_id, "vector": [0.2], "payload": {"id": new_id, "text": "alice"}},
        ]
    )
    adapter = _make_lance_adapter(table)

    assert asyncio.run(_rekey_native(adapter, "Entity_name", {old_id: new_id}))

    assert old_id not in table.rows
    assert list(table.rows) == [new_id]
    # The pre-existing new row survived untouched (no overwrite, no duplicate).
    assert table.rows[new_id]["vector"] == [0.2]


def test_native_lancedb_rekey_collapses_two_old_ids_to_one_target():
    """Downgrade case: Entity('x') and EntityType('x') both map to the bare id.
    One row moves, the other old row is dropped — never two rows with one id."""
    old_a = str(Entity.id_for("alice"))
    old_b = str(EntityType.id_for("alice"))
    bare = str(generate_node_id("alice"))
    table = _FakeLanceTable(
        [
            {"id": old_a, "vector": [0.1], "payload": {"id": old_a, "text": "alice"}},
            {"id": old_b, "vector": [0.2], "payload": {"id": old_b, "text": "alice"}},
        ]
    )
    adapter = _make_lance_adapter(table)

    assert asyncio.run(_rekey_native(adapter, "Entity_name", {old_a: bare, old_b: bare}))

    assert list(table.rows) == [bare]


def test_native_pgvector_rekey_merges_when_new_id_already_exists():
    """Same merge-safety for the SQL path: a pre-existing new-id row must make
    the old row a DELETE, not a duplicate-PK UPDATE that fails every startup."""
    from sqlalchemy import JSON, Column, MetaData, String, Table, select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    old_id = str(generate_node_id("Alice"))
    new_id = str(Entity.id_for("alice"))

    async def run():
        engine = create_async_engine("sqlite+aiosqlite://")
        metadata = MetaData()
        table = Table(
            "Entity_name",
            metadata,
            Column("id", String, primary_key=True),
            Column("payload", JSON),
            Column("vector", JSON),
        )
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(
                table.insert().values(
                    id=old_id, payload={"id": old_id, "text": "alice"}, vector=[0.1]
                )
            )
            await conn.execute(
                table.insert().values(
                    id=new_id, payload={"id": new_id, "text": "alice"}, vector=[0.2]
                )
            )

        class PGVectorAdapter:  # class name drives the dispatch
            async def get_table(self, collection):
                return table

            def get_async_session(self):
                return AsyncSession(engine)

        handled = await _rekey_native(PGVectorAdapter(), "Entity_name", {old_id: new_id})

        async with AsyncSession(engine) as session:
            rows = (await session.execute(select(table))).all()
        await engine.dispose()
        return handled, rows

    handled, rows = asyncio.run(run())
    assert handled
    assert len(rows) == 1
    assert rows[0].id == new_id
    assert rows[0].vector == [0.2]  # the pre-existing new row survived


# ── downgrade (reverse remap) ────────────────────────────────────────────────


def test_build_id_remap_reverse_targets_released_bare_scheme():
    from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
        build_id_remap_reverse,
    )

    entity_new = str(Entity.id_for("alice"))
    nodes = [
        (entity_new, {"name": "alice", "type": "Entity", "description": "a"}),
        ("chunk-1", {"name": "c", "type": "DocumentChunk"}),
    ]
    assert build_id_remap_reverse(nodes) == {entity_new: str(generate_node_id("alice"))}


def test_graph_upgrade_then_downgrade_roundtrip():
    """up() then down() restores the released bare scheme on the graph store."""
    from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
        build_id_remap_reverse,
    )

    graph = _FakeGraph()
    entity_bare = graph.add_old_node("Alice", "Entity")
    graph.edges = [("chunk-1", entity_bare, "contains", {})]

    asyncio.run(_run_graph_migration(graph))  # upgrade
    entity_new = str(Entity.id_for("alice"))
    assert entity_new in graph.nodes

    async def run_down():
        nodes, edges = await graph.get_graph_data()
        id_map = build_id_remap_reverse(nodes)
        properties_by_id = {nid: props for nid, props in nodes}
        await _migrate_graph(graph, id_map, properties_by_id, edges)

    asyncio.run(run_down())

    assert entity_new not in graph.nodes
    assert entity_bare in graph.nodes  # back on the released bare scheme
    assert ("chunk-1", entity_bare, "contains", {}) in graph.edges
