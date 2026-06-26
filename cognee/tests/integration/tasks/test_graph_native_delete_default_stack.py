"""Part 1 + Part 2 integration gate: graph-provenance delete/rollback on the REAL
default graph adapter (Ladybug/Kuzu) driven through the UnifiedStoreEngine.

These don't use the LLM or an embedding backend — they seed the graph directly
via the real Ladybug adapter and use a recording vector engine. That keeps them
fast and offline while still proving the thing the in-memory fakes can't: that
the real adapter's snapshots / lookups / mutations mesh with the planner and the
unified engine (the cross-component seam). The full add->cognify->delete e2e
(mocked LLM + real embeddings) lives in the slow_e2e suite.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LADYBUG, reason="ladybug not installed"),
]


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class RecordingVectorEngine:
    """Records vector deletes; optionally fails once on a named collection."""

    def __init__(self, fail_once_on: str | None = None):
        self.deleted: list[tuple[str, list[str]]] = []
        self.removed_tags: list[list[str]] = []
        self._fail_once_on = fail_once_on

    async def delete_data_points(self, collection: str, ids) -> None:
        if collection == self._fail_once_on:
            self._fail_once_on = None
            raise RuntimeError(f"injected vector failure on {collection}")
        self.deleted.append((collection, [str(i) for i in ids]))

    async def remove_belongs_to_set_tags(self, tags, node_ids=None) -> None:
        self.removed_tags.append(list(tags))

    def deleted_ids(self) -> set[str]:
        return {i for _collection, ids in self.deleted for i in ids}


def _engine(graph, vector):
    return UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vector,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )


async def test_data_item_delete_shared_node_survives(tmp_path):
    """Delete one data item's source ref: its unowned node + vector go; a node
    shared with another data item survives (real Ladybug)."""
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        ds = uuid4()
        key1 = make_source_ref_key(ds, uuid4())  # data item 1
        key2 = make_source_ref_key(ds, uuid4())  # data item 2

        alice, bob, berlin = _Ent(name="Alice"), _Ent(name="Bob"), _Ent(name="Berlin")
        await graph.add_nodes([alice, bob, berlin])
        await graph.add_edges([(str(alice.id), str(bob.id), "knows", {"edge_text": "knows"})])
        await graph.add_edges(
            [(str(alice.id), str(berlin.id), "lives_in", {"edge_text": "lives in"})]
        )

        # data1 owns Alice, Bob + the "knows" edge; data2 owns Alice (shared) +
        # Berlin + the "lives_in" edge.
        await graph.attach_node_source_refs([str(alice.id), str(bob.id)], [key1], str(uuid4()))
        await graph.attach_edge_source_refs(
            [EdgeIdentity(str(alice.id), str(bob.id), "knows")], [key1], str(uuid4())
        )
        await graph.attach_node_source_refs([str(alice.id), str(berlin.id)], [key2], str(uuid4()))
        await graph.attach_edge_source_refs(
            [EdgeIdentity(str(alice.id), str(berlin.id), "lives_in")], [key2], str(uuid4())
        )

        vector = RecordingVectorEngine()
        await _engine(graph, vector).delete_by_source_ref(key2)

        # Berlin was owned only by data2 -> gone; Alice survives (still data1),
        # Bob survives.
        remaining = await graph.get_node_delete_data([str(alice.id), str(bob.id), str(berlin.id)])
        assert str(berlin.id) not in remaining
        assert str(alice.id) in remaining
        assert remaining[str(alice.id)].source_ref_keys == [key1]  # key2 detached
        assert str(bob.id) in remaining

        # The "lives_in" edge is gone; "knows" survives.
        edges = await graph.get_edge_delete_data(
            [
                EdgeIdentity(str(alice.id), str(berlin.id), "lives_in"),
                EdgeIdentity(str(alice.id), str(bob.id), "knows"),
            ]
        )
        assert EdgeIdentity(str(alice.id), str(berlin.id), "lives_in") not in edges
        assert EdgeIdentity(str(alice.id), str(bob.id), "knows") in edges

        # Berlin's vector point was deleted; Alice's was not.
        assert str(berlin.id) in vector.deleted_ids()
        assert str(alice.id) not in vector.deleted_ids()
    finally:
        await graph.close()


async def test_rollback_keeps_artifact_a_prior_run_owns(tmp_path):
    """Over-deletion guard on the real adapter: a run that re-touched an already
    owned ref records no run ref, so rolling it back must not delete the node."""
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        key = make_source_ref_key(uuid4(), uuid4())  # SAME data item across runs
        run1, run2 = uuid4(), uuid4()

        node = _Ent(name="Shared")
        await graph.add_nodes([node])
        await graph.attach_node_source_refs([str(node.id)], [key], str(run1))
        await graph.attach_node_source_refs([str(node.id)], [key], str(run2))  # re-touch

        vector = RecordingVectorEngine()
        engine = _engine(graph, vector)

        # run2 introduced no new run ref -> rollback finds nothing -> node survives.
        await engine.rollback_by_pipeline_run_id(str(run2))
        snap = await graph.get_node_delete_data([str(node.id)])
        assert str(node.id) in snap
        assert snap[str(node.id)].source_ref_keys == [key]
        assert vector.deleted == []

        # run1 is the run that established ownership -> rolling it back deletes it.
        await engine.rollback_by_pipeline_run_id(str(run1))
        assert await graph.get_node_delete_data([str(node.id)]) == {}
    finally:
        await graph.close()


async def test_retry_converges_after_vector_failure(tmp_path):
    """An injected vector failure leaves the real graph untouched; retry completes."""
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        key = make_source_ref_key(uuid4(), uuid4())
        node = _Ent(name="Doomed")
        await graph.add_nodes([node])
        await graph.attach_node_source_refs([str(node.id)], [key], str(uuid4()))

        # Find the collection the planner will try to delete first.
        snap = await graph.get_node_delete_data([str(node.id)])
        node_type = snap[str(node.id)].node_type
        field = snap[str(node.id)].indexed_fields[0]
        collection = f"{node_type}_{field}"

        vector = RecordingVectorEngine(fail_once_on=collection)
        engine = _engine(graph, vector)

        with pytest.raises(RuntimeError, match="injected vector failure"):
            await engine.delete_by_source_ref(key)

        # Graph provenance is intact -> the node is still discoverable by ref.
        assert await graph.find_nodes_by_source_ref(key) == [str(node.id)]

        await engine.delete_by_source_ref(key)  # retry converges
        assert await graph.get_node_delete_data([str(node.id)]) == {}
    finally:
        await graph.close()
