"""Part 2 unit tests: UnifiedStoreEngine graph-native delete/rollback vs fakes.

These exercise the real planner + UnifiedStoreEngine orchestration against an
in-memory fake graph engine (implements the Part 0 provenance read/write
primitives) and a recording fake vector engine. They prove the acceptance
criteria without any real DB or Part 1 adapter:

- vectors deleted FIRST, refs removed from shared artifacts, only unowned
  artifacts deleted, no-candidate requests no-op;
- edge/triplet vector ids come from the Part 0 id helpers (EdgeType.id_for,
  generate_node_id), and orchestration never reads relational Node/Edge rows;
- an injected vector-delete failure raises with graph provenance untouched, and
  a retry converges to the clean final state;
- unsupported-capability reads propagate before any mutation.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
    UnsupportedProvenanceCapability,
    make_source_ref,
    make_source_run_ref,
)

DATASET_A = UUID("00000000-0000-0000-0000-0000000000a1")
DATASET_B = UUID("00000000-0000-0000-0000-0000000000b2")
DATA_1 = UUID("00000000-0000-0000-0000-0000000000d1")
DATA_2 = UUID("00000000-0000-0000-0000-0000000000d2")
RUN_OLD = UUID("00000000-0000-0000-0000-00000000ce01")
RUN_NEW = UUID("00000000-0000-0000-0000-00000000ce02")


class FakeVectorEngine:
    """Records delete_data_points calls; can be told to fail a collection once."""

    def __init__(self):
        self.deleted: list[tuple[str, list[str]]] = []
        self.fail_collection: str | None = None

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        if self.fail_collection is not None and collection_name == self.fail_collection:
            raise RuntimeError(f"injected vector delete failure on {collection_name}")
        self.deleted.append((collection_name, list(data_point_ids)))


class FakeProvenanceGraphEngine:
    """In-memory graph implementing the Part 0 provenance read/write primitives.

    Nodes are dicts keyed by node_id; edges are dicts keyed by EdgeIdentity. Each
    carries source_refs / source_run_refs / dataset_ids as mutable sets — the
    in-memory stand-in for what a real adapter stamps on the graph.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[EdgeIdentity, dict] = {}

    # -- capability flag ---------------------------------------------------
    def supports_graph_native_provenance(self) -> bool:
        return True

    # -- seeding helpers ---------------------------------------------------
    def add_node_record(
        self,
        node_id,
        node_type="Entity",
        indexed_fields=("name",),
        label=None,
        source_refs=(),
        source_run_refs=(),
        dataset_ids=(),
    ):
        self.nodes[str(node_id)] = {
            "node_type": node_type,
            "indexed_fields": tuple(indexed_fields),
            "label": label,
            "source_refs": set(source_refs),
            "source_run_refs": set(source_run_refs),
            "dataset_ids": set(str(d) for d in dataset_ids),
        }

    def add_edge_record(
        self,
        identity: EdgeIdentity,
        edge_retrieval_text=None,
        source_refs=(),
        source_run_refs=(),
        dataset_ids=(),
    ):
        self.edges[identity] = {
            "edge_retrieval_text": edge_retrieval_text,
            "source_refs": set(source_refs),
            "source_run_refs": set(source_run_refs),
            "dataset_ids": set(str(d) for d in dataset_ids),
        }

    # -- snapshot builders -------------------------------------------------
    def _node_snapshot(self, node_id, rec) -> NodeDeleteData:
        return NodeDeleteData(
            node_id=node_id,
            node_type=rec["node_type"],
            label=rec["label"],
            indexed_fields=rec["indexed_fields"],
            source_refs=tuple(rec["source_refs"]),
            source_run_refs=tuple(rec["source_run_refs"]),
            dataset_ids=tuple(rec["dataset_ids"]),
        )

    def _edge_snapshot(self, identity, rec) -> EdgeDeleteData:
        return EdgeDeleteData(
            identity=identity,
            edge_retrieval_text=rec["edge_retrieval_text"],
            source_refs=tuple(rec["source_refs"]),
            source_run_refs=tuple(rec["source_run_refs"]),
            dataset_ids=tuple(rec["dataset_ids"]),
        )

    # -- read primitives ---------------------------------------------------
    async def get_nodes_delete_data_by_source_ref(self, source_ref):
        return [
            self._node_snapshot(nid, r)
            for nid, r in self.nodes.items()
            if source_ref in r["source_refs"]
        ]

    async def get_edges_delete_data_by_source_ref(self, source_ref):
        return [
            self._edge_snapshot(i, r)
            for i, r in self.edges.items()
            if source_ref in r["source_refs"]
        ]

    async def get_nodes_delete_data_by_dataset_id(self, dataset_id):
        key = str(dataset_id)
        return [
            self._node_snapshot(nid, r) for nid, r in self.nodes.items() if key in r["dataset_ids"]
        ]

    async def get_edges_delete_data_by_dataset_id(self, dataset_id):
        key = str(dataset_id)
        return [self._edge_snapshot(i, r) for i, r in self.edges.items() if key in r["dataset_ids"]]

    async def get_nodes_delete_data_by_source_run_ref(self, source_run_ref):
        return [
            self._node_snapshot(nid, r)
            for nid, r in self.nodes.items()
            if source_run_ref in r["source_run_refs"]
        ]

    async def get_edges_delete_data_by_source_run_ref(self, source_run_ref):
        return [
            self._edge_snapshot(i, r)
            for i, r in self.edges.items()
            if source_run_ref in r["source_run_refs"]
        ]

    # -- write primitives --------------------------------------------------
    async def detach_provenance_refs_from_nodes(self, node_ids, property_key, refs):
        for nid in node_ids:
            rec = self.nodes.get(str(nid))
            if rec:
                rec[property_key] -= set(refs)

    async def detach_provenance_refs_from_edges(self, edges, property_key, refs):
        for identity in edges:
            rec = self.edges.get(identity)
            if rec:
                rec[property_key] -= set(refs)

    async def delete_nodes(self, node_ids):
        for nid in node_ids:
            self.nodes.pop(str(nid), None)

    async def delete_edges(self, edges):
        for identity in edges:
            self.edges.pop(identity, None)


class RaisingGraphEngine(FakeProvenanceGraphEngine):
    """Reads raise UnsupportedProvenanceCapability (an un-implemented backend)."""

    async def get_nodes_delete_data_by_source_ref(self, source_ref):
        raise UnsupportedProvenanceCapability("get_nodes_delete_data_by_source_ref")


def _engine(graph, vector) -> UnifiedStoreEngine:
    return UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vector,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )


# ---------------------------------------------------------------------------
# delete_by_source_ref
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_by_source_ref_deletes_unowned_detaches_shared():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    ref1 = make_source_ref(DATASET_A, DATA_1)
    ref2 = make_source_ref(DATASET_A, DATA_2)
    graph.add_node_record("solely", source_refs=(ref1,), dataset_ids=(DATASET_A,))
    graph.add_node_record("shared", source_refs=(ref1, ref2), dataset_ids=(DATASET_A,))

    result = _engine(graph, vector)
    res = await result.delete_by_source_ref(ref1)

    assert "solely" not in graph.nodes  # unowned → hard deleted
    assert "shared" in graph.nodes  # survives via ref2
    assert graph.nodes["shared"]["source_refs"] == {ref2}  # detached
    assert res.nodes_deleted == 1
    assert res.nodes_detached == 1
    # Vector delete targeted only the unowned node's collection.
    assert ("Entity_name", ["solely"]) in vector.deleted


@pytest.mark.asyncio
async def test_no_candidates_is_noop():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    res = await _engine(graph, vector).delete_by_source_ref(make_source_ref(DATASET_A, DATA_1))
    assert (res.nodes_deleted, res.edges_deleted, res.nodes_detached, res.edges_detached) == (
        0,
        0,
        0,
        0,
    )
    assert vector.deleted == []


# ---------------------------------------------------------------------------
# edge/triplet vector ids come from the Part 0 id helpers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_edge_vector_ids_use_part0_helpers():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    ref = make_source_ref(DATASET_A, DATA_1)
    identity = EdgeIdentity("src", "works_with", "tgt")
    graph.add_edge_record(
        identity, edge_retrieval_text="works with", source_refs=(ref,), dataset_ids=(DATASET_A,)
    )

    await _engine(graph, vector).delete_by_source_ref(ref)

    expected_edge_type_id = str(EdgeType.id_for("works with"))
    expected_triplet_id = str(generate_node_id("src" + "works_with" + "tgt"))
    assert ("EdgeType_relationship_name", [expected_edge_type_id]) in vector.deleted
    assert ("Triplet_text", [expected_triplet_id]) in vector.deleted
    assert identity not in graph.edges  # unowned edge deleted


# ---------------------------------------------------------------------------
# delete_by_dataset_id — cross-dataset preservation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_by_dataset_id_preserves_cross_dataset():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    a_ref = make_source_ref(DATASET_A, DATA_1)
    graph.add_node_record("a_only", source_refs=(a_ref,), dataset_ids=(DATASET_A,))
    graph.add_node_record("shared_ds", source_refs=(a_ref,), dataset_ids=(DATASET_A, DATASET_B))

    res = await _engine(graph, vector).delete_by_dataset_id(DATASET_A)

    assert "a_only" not in graph.nodes
    assert "shared_ds" in graph.nodes
    assert graph.nodes["shared_ds"]["dataset_ids"] == {str(DATASET_B)}
    assert res.nodes_deleted == 1 and res.nodes_detached == 1


# ---------------------------------------------------------------------------
# rollback_by_pipeline_run_id — only what the run solely introduced
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rollback_removes_only_run_introduced():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    old_run = make_source_run_ref(DATASET_A, RUN_OLD)
    new_run = make_source_run_ref(DATASET_A, RUN_NEW)
    src_ref = make_source_ref(DATASET_A, DATA_1)
    graph.add_node_record("only_new", source_run_refs=(new_run,), dataset_ids=(DATASET_A,))
    graph.add_node_record("both_runs", source_run_refs=(old_run, new_run), dataset_ids=(DATASET_A,))
    graph.add_node_record(
        "has_src", source_refs=(src_ref,), source_run_refs=(new_run,), dataset_ids=(DATASET_A,)
    )

    res = await _engine(graph, vector).rollback_by_pipeline_run_id(RUN_NEW, DATASET_A)

    assert "only_new" not in graph.nodes  # solely from the rolled-back run
    assert graph.nodes["both_runs"]["source_run_refs"] == {old_run}  # detached
    assert graph.nodes["has_src"]["source_run_refs"] == set()  # detached, kept by source ref
    assert res.nodes_deleted == 1 and res.nodes_detached == 2


@pytest.mark.asyncio
async def test_rollback_is_dataset_scoped():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    graph.add_node_record(
        "a", source_run_refs=(make_source_run_ref(DATASET_A, RUN_NEW),), dataset_ids=(DATASET_A,)
    )
    res = await _engine(graph, vector).rollback_by_pipeline_run_id(RUN_NEW, DATASET_B)
    assert "a" in graph.nodes  # different dataset → different run ref → no-op
    assert (res.nodes_deleted, res.nodes_detached) == (0, 0)


# ---------------------------------------------------------------------------
# retry-safety: injected vector failure leaves graph untouched, retry converges
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vector_failure_leaves_graph_untouched_then_retry_converges():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()
    ref = make_source_ref(DATASET_A, DATA_1)
    graph.add_node_record("solely", source_refs=(ref,), dataset_ids=(DATASET_A,))
    vector.fail_collection = "Entity_name"

    engine = _engine(graph, vector)
    with pytest.raises(RuntimeError, match="injected vector delete failure"):
        await engine.delete_by_source_ref(ref)

    # Graph provenance untouched — the node and its ref are still present.
    assert "solely" in graph.nodes
    assert graph.nodes["solely"]["source_refs"] == {ref}

    # Retry after clearing the failure converges to the clean final state.
    vector.fail_collection = None
    res = await engine.delete_by_source_ref(ref)
    assert "solely" not in graph.nodes
    assert res.nodes_deleted == 1


# ---------------------------------------------------------------------------
# unsupported-capability reads propagate before any mutation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unsupported_capability_propagates_before_mutation():
    graph = RaisingGraphEngine()
    vector = FakeVectorEngine()
    ref = make_source_ref(DATASET_A, DATA_1)
    graph.add_node_record("n", source_refs=(ref,), dataset_ids=(DATASET_A,))

    with pytest.raises(UnsupportedProvenanceCapability):
        await _engine(graph, vector).delete_by_source_ref(ref)

    assert vector.deleted == []  # no vector mutation before the read failed
    assert "n" in graph.nodes  # no graph mutation


@pytest.mark.asyncio
async def test_engine_advertises_support_with_provenance_graph():
    assert _engine(FakeProvenanceGraphEngine(), FakeVectorEngine()).supports_graph_native_delete()
