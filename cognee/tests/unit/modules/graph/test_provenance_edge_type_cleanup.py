"""Regression tests for EdgeType cleanup after graph-provenance deletes (issue #5473).

Two defects were reported against ``provenance_delete_planner``:

- **DETACH DELETE collateral (defect A)**: ``execute_source_ref_removal`` deletes
  unowned nodes before edges. Engines whose ``delete_nodes`` detaches
  (Ladybug/Neo4j ``DETACH DELETE``) take every incident edge with them — including
  edges other source refs still own — and those edges are not in the
  ``unowned_edges`` list, so their EdgeType artifacts were never pruned.
- **Raw-text vs normalized-id mismatch (defect B)**: the orphan check compared raw
  retrieval texts while the actual delete keys on ``EdgeType.id_for(text)``
  (normalized: lower-cased, spaces→``_``, apostrophes stripped). A surviving edge
  whose text differs only in case/spaces/apostrophes maps to the SAME point id,
  yet the raw comparison flagged the deleted spelling as orphaned and deleted the
  shared point out from under the survivor.

The fix compares candidate and remaining edges in the normalized id space and
snapshots the texts of edges incident to soon-to-be-detached nodes before
``delete_nodes`` runs. All tests below use in-memory fakes only — no graph or
vector backend, no LLM.
"""

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
)
from cognee.infrastructure.databases.unified.provenance_delete_planner import (
    execute_source_ref_removal,
)
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

pytestmark = pytest.mark.asyncio

EDGETYPE_COLLECTION = "EdgeType_relationship_name"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class DetachingFakeGraph:
    """In-memory graph whose ``delete_nodes`` mimics Ladybug/Neo4j DETACH DELETE.

    ``get_graph_data`` returns the same ``(nodes, edges)`` tuple shape as the real
    adapters: nodes as ``(id, properties)``, edges as
    ``(source_id, target_id, relationship_name, properties)``.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[EdgeIdentity, str] = {}  # edge -> edge_text
        self.deleted_node_ids: list[list[str]] = []
        self.deleted_edge_triples: list[EdgeIdentity] = []

    # -- setup helpers -------------------------------------------------------

    def add_node(self, node_id: str) -> None:
        self.nodes[node_id] = {}

    def add_edge(self, source_id, target_id, relationship_name, edge_text) -> EdgeIdentity:
        edge = EdgeIdentity(source_id, target_id, relationship_name)
        self.edges[edge] = edge_text
        return edge

    # -- contract used by the planner ---------------------------------------

    async def get_graph_data(self):
        nodes = [(node_id, dict(props)) for node_id, props in self.nodes.items()]
        edges = [
            (edge.source_id, edge.target_id, edge.relationship_name, {"edge_text": text})
            for edge, text in self.edges.items()
        ]
        return nodes, edges

    async def delete_nodes(self, node_ids):
        ids = set(node_ids)
        self.deleted_node_ids.append(list(node_ids))
        for node_id in ids:
            self.nodes.pop(node_id, None)
        # DETACH DELETE: every edge incident to a deleted node disappears too —
        # even edges other source refs still own.
        for edge in list(self.edges):
            if edge.source_id in ids or edge.target_id in ids:
                del self.edges[edge]

    async def delete_edge_triples(self, edges):
        for edge in edges:
            self.edges.pop(edge, None)
            self.deleted_edge_triples.append(edge)

    async def remove_node_source_refs(self, node_ids, source_ref_keys):
        return None

    async def remove_edge_source_refs(self, edges, source_ref_keys):
        return None

    async def remove_belongs_to_set_tags(self, tags):
        return None


class RecordingVectorEngine:
    """Records every ``delete_data_points`` call as ``(collection, ids)``."""

    def __init__(self):
        self.deleted: list[tuple[str, list[str]]] = []
        self._existing = {"Entity_name", "Triplet_text", EDGETYPE_COLLECTION}

    async def has_collection(self, collection: str) -> bool:
        return collection in self._existing

    async def delete_data_points(self, collection: str, ids: list[str]) -> None:
        self.deleted.append((collection, list(ids)))

    async def remove_belongs_to_set_tags(self, tags):
        return None

    def edgetype_deletes(self) -> list[str]:
        """Flat list of ids deleted from the EdgeType collection."""
        return [
            point_id
            for collection, ids in self.deleted
            if collection == EDGETYPE_COLLECTION
            for point_id in ids
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edgetype_node_id(edge_text: str) -> str:
    return str(EdgeType.id_for(get_edge_retrieval_text(edge_text, "is_a")))


def _edge_data(edge: EdgeIdentity, edge_text: str, refs: list[str]) -> EdgeDeleteData:
    return EdgeDeleteData(
        edge=edge,
        edge_text=edge_text,
        edge_properties={"edge_text": edge_text},
        source_ref_keys=list(refs),
        source_dataset_ids=[],
        source_run_ids=[],
        source_run_refs=[],
    )


def _node_data(node_id: str, refs: list[str]) -> NodeDeleteData:
    return NodeDeleteData(
        node_id=node_id,
        node_type="Entity",
        indexed_fields=["name"],
        node_properties={"name": node_id},
        source_ref_keys=list(refs),
        source_dataset_ids=[],
        source_run_ids=[],
        source_run_refs=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_id_for_normalization_is_the_deletion_key():
    """Pin the contract behind defect B: id_for normalizes case/spaces/apostrophes,
    so raw-text comparison can never decide point ownership."""
    assert str(EdgeType.id_for("Foo bar is a data.")) == str(EdgeType.id_for("foo bar is a data."))
    assert str(EdgeType.id_for("it's a match")) == str(EdgeType.id_for("its a match"))
    assert str(EdgeType.id_for("foo bar")) == str(EdgeType.id_for("foo_bar"))


@pytest.mark.parametrize(
    ("deleted_text", "survivor_text"),
    [
        ("Foo bar is a data.", "foo bar is a data."),  # case-only variant (the reported repro)
        ("It's a valid link.", "Its a valid link."),  # apostrophe-stripping variant
        ("shared link text", "shared_link_text"),  # space->underscore variant
    ],
)
async def test_variant_survivor_keeps_shared_edgetype_vector(deleted_text, survivor_text):
    """Defect B: deleting an edge whose text differs from a surviving edge only in
    normalized-equivalent spelling must NOT delete the shared EdgeType point."""
    # The premise itself: both spellings map to ONE point id.
    assert str(EdgeType.id_for(deleted_text)) == str(EdgeType.id_for(survivor_text))

    ref_deleted = make_ref()

    graph = DetachingFakeGraph()
    graph.add_node("n_a")
    graph.add_node("n_b")
    graph.add_node("n_c")
    deleted_edge = graph.add_edge("n_a", "n_b", "is_a", deleted_text)
    survivor = graph.add_edge("n_b", "n_c", "is_a", survivor_text)

    vector = RecordingVectorEngine()

    await execute_source_ref_removal(
        graph,
        vector,
        node_data={},
        edge_data={deleted_edge: _edge_data(deleted_edge, deleted_text, [ref_deleted])},
        refs_by_node={},
        refs_by_edge={deleted_edge: [ref_deleted]},
    )

    # The deleted edge itself is gone; the survivor stays.
    assert deleted_edge not in graph.edges
    assert survivor in graph.edges

    shared_id = _edgetype_node_id(deleted_text)
    assert shared_id not in vector.edgetype_deletes(), (
        "cleanup deleted an EdgeType point a surviving variant-spelling edge still uses"
    )


def make_ref() -> str:
    return f"source_ref:v1:{uuid4()}:{uuid4()}"


async def test_unmatched_deleted_text_is_pruned_by_normalized_id():
    """Defect B inverse: a deleted text with no normalized-equivalent survivor is
    still orphan-pruned, keyed by EdgeType.id_for (not raw text)."""
    ref = make_ref()
    other_ref = make_ref()

    graph = DetachingFakeGraph()
    graph.add_node("n_a")
    graph.add_node("n_b")
    graph.add_node("n_c")
    deleted_edge = graph.add_edge("n_a", "n_b", "is_a", "gone relationship text")
    survivor = graph.add_edge("n_b", "n_c", "is_a", "totally different text")

    vector = RecordingVectorEngine()

    await execute_source_ref_removal(
        graph,
        vector,
        node_data={},
        edge_data={deleted_edge: _edge_data(deleted_edge, "gone relationship text", [ref])},
        refs_by_node={},
        refs_by_edge={deleted_edge: [ref]},
    )

    orphan_id = _edgetype_node_id("gone relationship text")
    assert orphan_id in vector.edgetype_deletes()
    # Same id is removed from the graph (EdgeType node) and the vector collection.
    assert any(orphan_id in batch for batch in graph.deleted_node_ids)
    assert survivor in graph.edges
    assert _edgetype_node_id("totally different text") not in vector.edgetype_deletes()
    assert other_ref  # keep naming explicit


async def test_detach_deleted_collateral_edges_are_pruned():
    """Defect A: an edge incident to a deleted node is DETACH-DELETEd even though
    another source ref owns it. Its EdgeType artifacts must join the cleanup
    candidates (snapshot taken before delete_nodes), so no orphan remains."""
    ref_node_owner = make_ref()
    ref_edge_owner = make_ref()  # different document still owns the collateral edge

    graph = DetachingFakeGraph()
    graph.add_node("n_del")
    graph.add_node("n_x")
    graph.add_node("n_y")
    collateral = graph.add_edge("n_del", "n_x", "is_a", "collateral relationship text")
    keep = graph.add_edge("n_x", "n_y", "is_a", "surviving relationship text")

    vector = RecordingVectorEngine()

    await execute_source_ref_removal(
        graph,
        vector,
        node_data={"n_del": _node_data("n_del", [ref_node_owner])},
        edge_data={
            collateral: _edge_data(collateral, "collateral relationship text", [ref_edge_owner])
        },
        refs_by_node={"n_del": [ref_node_owner]},
        refs_by_edge={},
    )

    # The node is gone and took the collateral edge with it (DETACH DELETE).
    assert "n_del" not in graph.nodes
    assert collateral not in graph.edges
    assert keep in graph.edges

    collateral_id = _edgetype_node_id("collateral relationship text")
    assert collateral_id in vector.edgetype_deletes(), (
        "EdgeType artifacts of a DETACH-DELETEd edge were left orphaned"
    )
    assert any(collateral_id in batch for batch in graph.deleted_node_ids)
    # The genuinely surviving edge keeps its point.
    assert _edgetype_node_id("surviving relationship text") not in vector.edgetype_deletes()


async def test_blank_edge_texts_and_empty_remaining_stay_idempotent():
    """Fully blank texts (edge_text AND relationship_name) are skipped; a fully
    emptied graph still prunes the last edge's EdgeType artifacts."""
    ref = make_ref()

    graph = DetachingFakeGraph()
    graph.add_node("n_a")
    graph.add_node("n_b")
    # Blank edge_text alone would fall back to relationship_name (existing
    # get_edge_retrieval_text semantics), so both must be blank to be skipped.
    deleted_edge = graph.add_edge("n_a", "n_b", "   ", "   ")

    vector = RecordingVectorEngine()

    await execute_source_ref_removal(
        graph,
        vector,
        node_data={},
        edge_data={deleted_edge: _edge_data(deleted_edge, "   ", [ref])},
        refs_by_node={},
        refs_by_edge={deleted_edge: [ref]},
    )

    assert deleted_edge not in graph.edges
    assert vector.edgetype_deletes() == []

    # Empty remaining graph: the deleted text's point is pruned (no survivor).
    graph2 = DetachingFakeGraph()
    graph2.add_node("n_a")
    graph2.add_node("n_b")
    last_edge = graph2.add_edge("n_a", "n_b", "is_a", "last text standing")
    vector2 = RecordingVectorEngine()

    await execute_source_ref_removal(
        graph2,
        vector2,
        node_data={},
        edge_data={last_edge: _edge_data(last_edge, "last text standing", [ref])},
        refs_by_node={},
        refs_by_edge={last_edge: [ref]},
    )

    assert _edgetype_node_id("last text standing") in vector2.edgetype_deletes()
