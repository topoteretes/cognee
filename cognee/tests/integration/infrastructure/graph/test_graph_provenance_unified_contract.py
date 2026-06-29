"""Backend-neutral graph-provenance contract through UnifiedStoreEngine.

Add a provider to ``graph_provenance_unified_engine`` when a graph backend
claims support for graph-provenance delete/rollback. These tests use a real
graph adapter and a recording vector engine so failures point at graph
provenance behavior or planner composition, not embedding/vector setup.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import NodeSet
from cognee.modules.engine.utils import generate_node_id

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = pytest.mark.asyncio


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class RecordingVectorEngine:
    """Records vector-side mutations and can fail once on one collection."""

    def __init__(self, fail_once_on: str | None = None):
        self.deleted: list[tuple[str, list[str]]] = []
        self.removed_tags: list[tuple[list[str], list[str] | None]] = []
        self._fail_once_on = fail_once_on

    async def delete_data_points(self, collection: str, ids) -> None:
        if collection == self._fail_once_on:
            self._fail_once_on = None
            raise RuntimeError(f"injected vector failure on {collection}")
        self.deleted.append((collection, [str(item_id) for item_id in ids]))

    async def remove_belongs_to_set_tags(self, tags, node_ids=None) -> None:
        scoped_ids = [str(node_id) for node_id in node_ids] if node_ids else None
        self.removed_tags.append((list(tags), scoped_ids))

    def deleted_ids(self) -> set[str]:
        return {item_id for _collection, ids in self.deleted for item_id in ids}

    def deleted_ids_by_collection(self, collection: str) -> set[str]:
        return {
            item_id
            for deleted_collection, ids in self.deleted
            if deleted_collection == collection
            for item_id in ids
        }


@pytest_asyncio.fixture(params=["ladybug+recording_vector"])
async def graph_provenance_unified_engine(request, tmp_path):
    if request.param == "ladybug+recording_vector":
        if not HAS_LADYBUG:
            pytest.skip("ladybug not installed")
        graph = LadybugAdapter(str(tmp_path / "graph_db"))
    else:
        raise AssertionError(f"Unknown graph provenance provider: {request.param}")

    vector = RecordingVectorEngine()
    engine = UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vector,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )

    try:
        yield SimpleNamespace(graph=graph, vector=vector, engine=engine)
    finally:
        await graph.close()


def _new_engine(graph, vector):
    return UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vector,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )


async def test_delete_by_source_ref_removes_unowned_and_detaches_shared(
    graph_provenance_unified_engine,
):
    graph = graph_provenance_unified_engine.graph
    vector = graph_provenance_unified_engine.vector
    engine = graph_provenance_unified_engine.engine
    ds = uuid4()
    key1 = make_source_ref_key(ds, uuid4())
    key2 = make_source_ref_key(ds, uuid4())

    alice, bob, berlin = _Ent(name="Alice"), _Ent(name="Bob"), _Ent(name="Berlin")
    await graph.add_nodes([alice, bob, berlin])
    await graph.add_edges([(str(alice.id), str(bob.id), "knows", {"edge_text": "knows"})])
    await graph.add_edges([(str(alice.id), str(berlin.id), "lives_in", {"edge_text": "lives in"})])

    knows = EdgeIdentity(str(alice.id), str(bob.id), "knows")
    lives_in = EdgeIdentity(str(alice.id), str(berlin.id), "lives_in")

    await graph.attach_node_source_refs([str(alice.id), str(bob.id)], [key1], str(uuid4()))
    await graph.attach_edge_source_refs([knows], [key1], str(uuid4()))
    await graph.attach_node_source_refs([str(alice.id), str(berlin.id)], [key2], str(uuid4()))
    await graph.attach_edge_source_refs([lives_in], [key2], str(uuid4()))

    await engine.delete_by_source_ref(key2)

    remaining = await graph.get_node_delete_data([str(alice.id), str(bob.id), str(berlin.id)])
    assert str(berlin.id) not in remaining
    assert remaining[str(alice.id)].source_ref_keys == [key1]
    assert str(bob.id) in remaining

    edges = await graph.get_edge_delete_data([lives_in, knows])
    assert lives_in not in edges
    assert knows in edges

    assert str(berlin.id) in vector.deleted_ids()
    assert str(alice.id) not in vector.deleted_ids()
    triplet_id = str(
        generate_node_id(lives_in.source_id + lives_in.relationship_name + lives_in.target_id)
    )
    assert triplet_id in vector.deleted_ids_by_collection("Triplet_text")


async def test_delete_by_dataset_id_preserves_cross_dataset_artifacts(
    graph_provenance_unified_engine,
):
    graph = graph_provenance_unified_engine.graph
    engine = graph_provenance_unified_engine.engine
    dataset_a, dataset_b = uuid4(), uuid4()
    ref_a = make_source_ref_key(dataset_a, uuid4())
    ref_b = make_source_ref_key(dataset_b, uuid4())
    run = str(uuid4())

    only_a = _Ent(name="Only A")
    only_b = _Ent(name="Only B")
    shared = _Ent(name="Shared")
    await graph.add_nodes([only_a, only_b, shared])
    await graph.attach_node_source_refs([str(only_a.id)], [ref_a], run)
    await graph.attach_node_source_refs([str(only_b.id)], [ref_b], run)
    await graph.attach_node_source_refs([str(shared.id)], [ref_a, ref_b], run)

    await engine.delete_by_dataset_id(str(dataset_a))

    snaps = await graph.get_node_delete_data([str(only_a.id), str(only_b.id), str(shared.id)])
    assert str(only_a.id) not in snaps
    assert snaps[str(only_b.id)].source_ref_keys == [ref_b]
    assert snaps[str(shared.id)].source_ref_keys == [ref_b]


async def test_rollback_by_pipeline_run_id_preserves_prior_owner(
    graph_provenance_unified_engine,
):
    graph = graph_provenance_unified_engine.graph
    engine = graph_provenance_unified_engine.engine
    key = make_source_ref_key(uuid4(), uuid4())
    run1, run2 = uuid4(), uuid4()
    node = _Ent(name="Shared")
    await graph.add_nodes([node])

    await graph.attach_node_source_refs([str(node.id)], [key], str(run1))
    await graph.attach_node_source_refs([str(node.id)], [key], str(run2))

    await engine.rollback_by_pipeline_run_id(str(run2))
    snap = await graph.get_node_delete_data([str(node.id)])
    assert snap[str(node.id)].source_ref_keys == [key]
    assert graph_provenance_unified_engine.vector.deleted == []

    await engine.rollback_by_pipeline_run_id(str(run1))
    assert await graph.get_node_delete_data([str(node.id)]) == {}


async def test_vector_failure_leaves_refs_discoverable_and_retry_converges(tmp_path):
    if not HAS_LADYBUG:
        pytest.skip("ladybug not installed")

    graph = LadybugAdapter(str(tmp_path / "graph_db"))
    try:
        key = make_source_ref_key(uuid4(), uuid4())
        node = _Ent(name="Doomed")
        await graph.add_nodes([node])
        await graph.attach_node_source_refs([str(node.id)], [key], str(uuid4()))

        snap = await graph.get_node_delete_data([str(node.id)])
        node_type = snap[str(node.id)].node_type
        field = snap[str(node.id)].indexed_fields[0]
        collection = f"{node_type}_{field}"

        vector = RecordingVectorEngine(fail_once_on=collection)
        engine = _new_engine(graph, vector)

        with pytest.raises(RuntimeError, match="injected vector failure"):
            await engine.delete_by_source_ref(key)

        assert await graph.find_nodes_by_source_ref(key) == [str(node.id)]

        await engine.delete_by_source_ref(key)
        assert await graph.get_node_delete_data([str(node.id)]) == {}
    finally:
        await graph.close()


async def test_deleting_nodeset_triggers_graph_and_vector_detag_cleanup(
    graph_provenance_unified_engine,
):
    graph = graph_provenance_unified_engine.graph
    vector = graph_provenance_unified_engine.vector
    engine = graph_provenance_unified_engine.engine
    key_nodeset = make_source_ref_key(uuid4(), uuid4())
    key_survivor = make_source_ref_key(uuid4(), uuid4())

    node_set = NodeSet(name="Dev")
    survivor = _Ent(name="Survivor", belongs_to_set=["Dev"])
    await graph.add_nodes([node_set, survivor])
    await graph.attach_node_source_refs([str(node_set.id)], [key_nodeset], str(uuid4()))
    await graph.attach_node_source_refs([str(survivor.id)], [key_survivor], str(uuid4()))

    await engine.delete_by_source_ref(key_nodeset)

    snaps = await graph.get_node_delete_data([str(node_set.id), str(survivor.id)])
    assert str(node_set.id) not in snaps
    assert snaps[str(survivor.id)].node_properties.get("belongs_to_set") == []
    assert (["Dev"], None) in vector.removed_tags
