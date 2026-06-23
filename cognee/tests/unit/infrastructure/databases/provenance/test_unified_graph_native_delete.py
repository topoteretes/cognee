"""Unit tests for the Part 2 graph-native delete/rollback spine.

These exercise ``UnifiedStoreEngine.delete_by_source_ref`` /
``delete_by_dataset_id`` / ``rollback_by_pipeline_run_id`` and the planner
(`execute_source_ref_removal`) against an in-memory fake graph engine that
implements Lazar's Part 0 provenance contract (parseable refs, ``EdgeIdentity``,
the two-step discovery API, the metadata marker, and ``None``-returning delete
methods) plus a recording fake vector engine.

The fakes store provenance as ``source_ref_keys`` lists on each node/edge and
derive ``source_dataset_ids`` / ``source_run_ids`` / ``source_run_refs`` on
attach, exactly as a real adapter would. The tests prove the semantics from the
spec:

- unowned hard-delete + shared detach on ``delete_by_source_ref``;
- cross-dataset preservation on ``delete_by_dataset_id``;
- rollback removes only run-introduced artifacts;
- vectors-first ordering + retry convergence after an injected vector failure;
- unsupported-capability propagation;
- no-candidate no-op.
"""

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
    get_dataset_id_from_source_ref_key,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.infrastructure.databases.unified import UnifiedStoreEngine
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# In-memory fakes implementing Lazar's contract.
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, node_id, node_type, indexed_fields, properties):
        self.node_id = node_id
        self.node_type = node_type
        self.indexed_fields = list(indexed_fields)
        self.properties = dict(properties)
        # Provenance state — kept as parallel lists like a real adapter.
        self.source_ref_keys: list[str] = []
        self.source_run_refs: list[str] = []


class _FakeEdge:
    def __init__(self, edge: EdgeIdentity, edge_text, properties=None):
        self.edge = edge
        self.edge_text = edge_text
        self.properties = dict(properties or {})
        self.source_ref_keys: list[str] = []
        self.source_run_refs: list[str] = []


class FakeVectorEngine:
    """Records delete_data_points / remove_belongs_to_set_tags calls.

    ``fail_on_collection`` injects a single failure for the named collection to
    prove the planner deletes vectors before any graph mutation (so a retry
    converges).
    """

    def __init__(self, fail_on_collection: str | None = None):
        self.deleted: list[tuple[str, list[str]]] = []
        self.removed_tags: list[list[str]] = []
        self._fail_on_collection = fail_on_collection
        self._existing_collections = {
            "Entity_name",
            "NodeSet_name",
            "EdgeType_relationship_name",
        }

    async def delete_data_points(self, collection: str, ids: list[str]) -> None:
        if collection == self._fail_on_collection:
            # Arm once, then let the retry succeed.
            self._fail_on_collection = None
            raise RuntimeError(f"injected vector failure on {collection}")
        if collection not in self._existing_collections:
            # Mirror the real "Triplet collection may not exist" case.
            raise RuntimeError(f"collection {collection} does not exist")
        self.deleted.append((collection, list(ids)))

    async def remove_belongs_to_set_tags(self, tags: list[str]) -> None:
        self.removed_tags.append(list(tags))


class FakeProvenanceGraphEngine:
    """In-memory graph implementing Lazar's Part 0 provenance methods."""

    def __init__(self, supports_provenance: bool = True):
        self._supports = supports_provenance
        self._metadata: dict[str, str] = {}
        self.nodes: dict[str, _FakeNode] = {}
        self.edges: dict[EdgeIdentity, _FakeEdge] = {}

    def _guard(self):
        if not self._supports:
            raise UnsupportedProvenanceCapability()

    # -- test setup helpers (not part of the contract) ----------------------

    def add_node(self, node_id, node_type, indexed_fields, properties=None):
        node = _FakeNode(node_id, node_type, indexed_fields, properties or {})
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id, target_id, relationship_name, edge_text, properties=None):
        edge = EdgeIdentity(source_id, target_id, relationship_name)
        self.edges[edge] = _FakeEdge(edge, edge_text, properties)
        return edge

    # -- marker --------------------------------------------------------------

    async def get_graph_metadata(self) -> dict[str, str]:
        self._guard()
        return dict(self._metadata)

    async def set_graph_metadata(self, metadata: dict[str, str]) -> None:
        self._guard()
        self._metadata.update(metadata)

    async def is_empty(self) -> bool:
        return not self.nodes and not self.edges

    # -- write ---------------------------------------------------------------

    async def attach_node_source_refs(self, node_ids, source_ref_keys, pipeline_run_id=None):
        self._guard()
        for node_id in node_ids:
            node = self.nodes[node_id]
            for key in source_ref_keys:
                if key not in node.source_ref_keys:
                    node.source_ref_keys.append(key)
                if pipeline_run_id is not None:
                    run_ref = make_source_run_ref(_as_uuid(pipeline_run_id), key)
                    if run_ref not in node.source_run_refs:
                        node.source_run_refs.append(run_ref)

    async def attach_edge_source_refs(self, edges, source_ref_keys, pipeline_run_id=None):
        self._guard()
        for edge in edges:
            row = self.edges[edge]
            for key in source_ref_keys:
                if key not in row.source_ref_keys:
                    row.source_ref_keys.append(key)
                if pipeline_run_id is not None:
                    run_ref = make_source_run_ref(_as_uuid(pipeline_run_id), key)
                    if run_ref not in row.source_run_refs:
                        row.source_run_refs.append(run_ref)

    async def remove_node_source_refs(self, node_ids, source_ref_keys):
        self._guard()
        removed = set(source_ref_keys)
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue  # idempotent: already gone
            node.source_ref_keys = [k for k in node.source_ref_keys if k not in removed]
            node.source_run_refs = [
                r for r in node.source_run_refs if _source_ref_of(r) not in removed
            ]

    async def remove_edge_source_refs(self, edges, source_ref_keys):
        self._guard()
        removed = set(source_ref_keys)
        for edge in edges:
            row = self.edges.get(edge)
            if row is None:
                continue
            row.source_ref_keys = [k for k in row.source_ref_keys if k not in removed]
            row.source_run_refs = [
                r for r in row.source_run_refs if _source_ref_of(r) not in removed
            ]

    async def delete_nodes(self, node_ids):
        self._guard()
        for node_id in node_ids:
            self.nodes.pop(node_id, None)  # idempotent

    async def delete_edge_triples(self, edges):
        self._guard()
        for edge in edges:
            self.edges.pop(edge, None)  # idempotent

    # -- read ----------------------------------------------------------------

    async def get_node_delete_data(self, node_ids):
        self._guard()
        out: dict[str, NodeDeleteData] = {}
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            out[node_id] = NodeDeleteData(
                node_id=node_id,
                node_type=node.node_type,
                indexed_fields=list(node.indexed_fields),
                node_properties=dict(node.properties),
                source_ref_keys=list(node.source_ref_keys),
                source_dataset_ids=_dataset_ids(node.source_ref_keys),
                source_run_ids=_run_ids(node.source_run_refs),
                source_run_refs=list(node.source_run_refs),
            )
        return out

    async def get_edge_delete_data(self, edges):
        self._guard()
        out: dict[EdgeIdentity, EdgeDeleteData] = {}
        for edge in edges:
            row = self.edges.get(edge)
            if row is None:
                continue
            out[edge] = EdgeDeleteData(
                edge=edge,
                edge_text=row.edge_text,
                edge_properties=dict(row.properties),
                source_ref_keys=list(row.source_ref_keys),
                source_dataset_ids=_dataset_ids(row.source_ref_keys),
                source_run_ids=_run_ids(row.source_run_refs),
                source_run_refs=list(row.source_run_refs),
            )
        return out

    async def find_nodes_by_source_ref(self, source_ref_key):
        self._guard()
        return [nid for nid, node in self.nodes.items() if source_ref_key in node.source_ref_keys]

    async def find_edges_by_source_ref(self, source_ref_key):
        self._guard()
        return [e for e, row in self.edges.items() if source_ref_key in row.source_ref_keys]

    async def find_node_source_refs_by_dataset(self, dataset_id):
        self._guard()
        out: dict[str, list[str]] = {}
        for nid, node in self.nodes.items():
            keys = [k for k in node.source_ref_keys if _dataset_of(k) == str(dataset_id)]
            if keys:
                out[nid] = keys
        return out

    async def find_edge_source_refs_by_dataset(self, dataset_id):
        self._guard()
        out: dict[EdgeIdentity, list[str]] = {}
        for e, row in self.edges.items():
            keys = [k for k in row.source_ref_keys if _dataset_of(k) == str(dataset_id)]
            if keys:
                out[e] = keys
        return out

    async def find_node_source_refs_by_pipeline_run(self, pipeline_run_id):
        self._guard()
        out: dict[str, list[str]] = {}
        for nid, node in self.nodes.items():
            keys = [
                _source_ref_of(r)
                for r in node.source_run_refs
                if _run_of(r) == str(pipeline_run_id)
            ]
            if keys:
                out[nid] = keys
        return out

    async def find_edge_source_refs_by_pipeline_run(self, pipeline_run_id):
        self._guard()
        out: dict[EdgeIdentity, list[str]] = {}
        for e, row in self.edges.items():
            keys = [
                _source_ref_of(r) for r in row.source_run_refs if _run_of(r) == str(pipeline_run_id)
            ]
            if keys:
                out[e] = keys
        return out

    async def get_graph_data(self):
        self._guard()
        nodes = [(nid, node.properties) for nid, node in self.nodes.items()]
        edges = [
            (e.source_id, e.target_id, e.relationship_name, {"edge_text": row.edge_text})
            for e, row in self.edges.items()
        ]
        return nodes, edges

    async def remove_belongs_to_set_tags(self, tags):
        self._guard()
        self.removed_tags = getattr(self, "removed_tags", [])
        self.removed_tags.append(list(tags))


# ---------------------------------------------------------------------------
# parseable-ref helpers (mirror what a real adapter derives on attach).
# ---------------------------------------------------------------------------


def _as_uuid(value):
    from uuid import UUID

    return value if hasattr(value, "hex") else UUID(str(value))


def _dataset_of(source_ref_key: str) -> str:
    # "source_ref:v1:{dataset}:{data}"
    return source_ref_key.split(":", 3)[2]


def _source_ref_of(source_run_ref: str) -> str:
    # "source_run_ref:v1:{run}:source_ref:v1:{dataset}:{data}"
    return source_run_ref.split(":", 3)[3]


def _run_of(source_run_ref: str) -> str:
    return source_run_ref.split(":", 3)[2]


def _dataset_ids(source_ref_keys):
    return sorted({_dataset_of(k) for k in source_ref_keys})


def _run_ids(source_run_refs):
    return sorted({_run_of(r) for r in source_run_refs})


def _build_engine(graph, vector):
    return UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vector,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_delete_by_source_ref_unowned_delete_and_shared_detach():
    """Nodes owned only by the deleted ref are hard-deleted; shared nodes detach."""
    dataset_a = uuid4()
    dataset_b = uuid4()
    data_a = uuid4()
    data_b = uuid4()
    run = uuid4()

    ref_a = make_source_ref_key(dataset_a, data_a)
    ref_b = make_source_ref_key(dataset_b, data_b)

    graph = FakeProvenanceGraphEngine()
    owned = graph.add_node("n_owned", "Entity", ["name"], {"name": "Germany"})
    shared = graph.add_node("n_shared", "Entity", ["name"], {"name": "Europe"})
    edge_owned = graph.add_edge("n_owned", "n_shared", "located_in", "located in")

    await graph.attach_node_source_refs(["n_owned"], [ref_a], run)
    await graph.attach_node_source_refs(["n_shared"], [ref_a, ref_b], run)
    await graph.attach_edge_source_refs([edge_owned], [ref_a], run)

    vector = FakeVectorEngine()
    engine = _build_engine(graph, vector)

    await engine.delete_by_source_ref(ref_a)

    # Owned node gone; shared node survives with only ref_b remaining.
    assert "n_owned" not in graph.nodes
    assert "n_shared" in graph.nodes
    assert graph.nodes["n_shared"].source_ref_keys == [ref_b]
    # Owned edge gone.
    assert edge_owned not in graph.edges

    # Vectors deleted for the unowned node + edge, not the shared node.
    deleted_collections = {c for c, _ in vector.deleted}
    assert ("Entity_name", ["n_owned"]) in vector.deleted
    assert all(ids != ["n_shared"] for _, ids in vector.deleted)
    assert "EdgeType_relationship_name" in deleted_collections
    assert owned is not None and shared is not None  # keep references explicit


async def test_delete_by_dataset_id_preserves_cross_dataset_artifacts():
    """Removing dataset A's refs leaves dataset-B-owned artifacts intact."""
    dataset_a = uuid4()
    dataset_b = uuid4()
    ref_a = make_source_ref_key(dataset_a, uuid4())
    ref_b = make_source_ref_key(dataset_b, uuid4())
    run = uuid4()

    graph = FakeProvenanceGraphEngine()
    graph.add_node("only_a", "Entity", ["name"], {"name": "A-only"})
    graph.add_node("only_b", "Entity", ["name"], {"name": "B-only"})
    graph.add_node("shared", "Entity", ["name"], {"name": "shared"})

    await graph.attach_node_source_refs(["only_a"], [ref_a], run)
    await graph.attach_node_source_refs(["only_b"], [ref_b], run)
    await graph.attach_node_source_refs(["shared"], [ref_a, ref_b], run)

    vector = FakeVectorEngine()
    engine = _build_engine(graph, vector)

    await engine.delete_by_dataset_id(str(dataset_a))

    # A-owned-only node is gone; B-only untouched; shared survives with ref_b.
    assert "only_a" not in graph.nodes
    assert "only_b" in graph.nodes
    assert graph.nodes["only_b"].source_ref_keys == [ref_b]
    assert "shared" in graph.nodes
    assert graph.nodes["shared"].source_ref_keys == [ref_b]

    assert ("Entity_name", ["only_a"]) in vector.deleted
    assert all(ids != ["only_b"] for _, ids in vector.deleted)


async def test_rollback_removes_only_run_introduced_artifacts():
    """Rollback removes refs the run attached; artifacts owned otherwise survive."""
    dataset = uuid4()
    ref_run1 = make_source_ref_key(dataset, uuid4())
    ref_run2 = make_source_ref_key(dataset, uuid4())
    run_1 = uuid4()
    run_2 = uuid4()

    graph = FakeProvenanceGraphEngine()
    graph.add_node("only_run2", "Entity", ["name"], {"name": "new"})
    graph.add_node("shared", "Entity", ["name"], {"name": "old"})

    # run_1 attached ref_run1 to "shared"; run_2 later attached ref_run2 to both
    # "shared" and "only_run2". Rolling back run_2 removes only ref_run2.
    await graph.attach_node_source_refs(["shared"], [ref_run1], run_1)
    await graph.attach_node_source_refs(["shared"], [ref_run2], run_2)
    await graph.attach_node_source_refs(["only_run2"], [ref_run2], run_2)

    vector = FakeVectorEngine()
    engine = _build_engine(graph, vector)

    await engine.rollback_by_pipeline_run_id(str(run_2))

    # only_run2's sole ref came in via run_2 -> unowned -> deleted.
    assert "only_run2" not in graph.nodes
    # shared still owns ref_run1 (attached by run_1) -> survives, detached of ref_run2.
    assert "shared" in graph.nodes
    assert graph.nodes["shared"].source_ref_keys == [ref_run1]

    assert ("Entity_name", ["only_run2"]) in vector.deleted


async def test_vectors_deleted_before_graph_mutation_and_retry_converges():
    """An injected vector failure leaves graph intact; the retry completes."""
    dataset = uuid4()
    ref = make_source_ref_key(dataset, uuid4())
    run = uuid4()

    graph = FakeProvenanceGraphEngine()
    graph.add_node("n1", "Entity", ["name"], {"name": "x"})
    await graph.attach_node_source_refs(["n1"], [ref], run)

    vector = FakeVectorEngine(fail_on_collection="Entity_name")
    engine = _build_engine(graph, vector)

    # First attempt fails inside the vector delete (before any graph mutation).
    with pytest.raises(RuntimeError, match="injected vector failure"):
        await engine.delete_by_source_ref(ref)

    # Graph provenance is untouched: the node and its ref still exist.
    assert "n1" in graph.nodes
    assert graph.nodes["n1"].source_ref_keys == [ref]

    # Retry converges now that the injected failure is disarmed.
    await engine.delete_by_source_ref(ref)
    assert "n1" not in graph.nodes
    assert ("Entity_name", ["n1"]) in vector.deleted


async def test_unsupported_capability_propagates():
    """A backend without provenance raises UnsupportedProvenanceCapability."""
    graph = FakeProvenanceGraphEngine(supports_provenance=False)
    vector = FakeVectorEngine()
    engine = _build_engine(graph, vector)

    with pytest.raises(UnsupportedProvenanceCapability):
        await engine.delete_by_source_ref(make_source_ref_key(uuid4(), uuid4()))


async def test_no_candidate_is_a_noop():
    """No matching artifacts -> no graph or vector mutation."""
    graph = FakeProvenanceGraphEngine()
    graph.add_node("survivor", "Entity", ["name"], {"name": "keep"})
    other_ref = make_source_ref_key(uuid4(), uuid4())
    await graph.attach_node_source_refs(["survivor"], [other_ref], uuid4())

    vector = FakeVectorEngine()
    engine = _build_engine(graph, vector)

    unrelated_ref = make_source_ref_key(uuid4(), uuid4())
    await engine.delete_by_source_ref(unrelated_ref)

    assert "survivor" in graph.nodes
    assert graph.nodes["survivor"].source_ref_keys == [other_ref]
    assert vector.deleted == []


async def test_supports_graph_native_delete_requires_both_capabilities():
    graph = FakeProvenanceGraphEngine()
    vector = FakeVectorEngine()

    assert _build_engine(graph, vector).supports_graph_native_delete() is True

    graph_only = UnifiedStoreEngine(graph_engine=graph, capabilities=EngineCapability.GRAPH)
    assert graph_only.supports_graph_native_delete() is False


async def test_orphaned_nodeset_tags_stripped_on_delete():
    """Deleting a uniquely-owned NodeSet strips its tag from surviving rows."""
    dataset = uuid4()
    ref = make_source_ref_key(dataset, uuid4())
    run = uuid4()

    graph = FakeProvenanceGraphEngine()
    graph.add_node("ns", "NodeSet", ["name"], {"name": "my_nodeset"})
    await graph.attach_node_source_refs(["ns"], [ref], run)

    vector = FakeVectorEngine()
    engine = _build_engine(graph, vector)

    await engine.delete_by_source_ref(ref)

    assert "ns" not in graph.nodes
    assert getattr(graph, "removed_tags", []) == [["my_nodeset"]]
    assert vector.removed_tags == [["my_nodeset"]]


# Sanity: the parseable-ref helpers used by the fakes agree with the contract.
async def test_fake_dataset_helper_matches_contract():
    dataset = uuid4()
    ref = make_source_ref_key(dataset, uuid4())
    assert _dataset_of(ref) == str(dataset)
    assert str(get_dataset_id_from_source_ref_key(ref)) == _dataset_of(ref)


# Reference the imports used only inside helpers so linters keep them.
_ = (EdgeType, generate_node_id, get_edge_retrieval_text)
