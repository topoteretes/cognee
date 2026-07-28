from unittest.mock import MagicMock

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.graph_enrichment import (
    canonicalize_graphs,
    extend_graph_with_ontology,
)
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


class _CanonicalizeStubResolver(BaseOntologyResolver):
    """Resolver with deterministic matches for canonicalize_graphs tests."""

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "gadget":
            root = AttachedOntologyNode("https://example.test/ontology#gadget", "classes")
            return [root], [], root
        if node_type == "individuals" and node_name == "widget":
            root = AttachedOntologyNode(
                "https://example.test/ontology#widget_canonical", "individuals"
            )
            related = AttachedOntologyNode("https://example.test/ontology#related", "individuals")
            return [root, related], [], root
        if node_type == "individuals" and node_name == "alpha":
            root = AttachedOntologyNode("https://example.test/ontology#alpha", "individuals")
            return [root], [], root
        return [], [], None


def _make_chunk():
    chunk = MagicMock()
    chunk.importance_weight = 0.5
    chunk.belongs_to_set = []
    return chunk


def test_canonicalize_graphs_passes_through_when_resolver_is_none():
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Widget", type="Gadget", description="desc")],
        edges=[],
    )
    copied, matches = canonicalize_graphs([graph], [_make_chunk()], None)
    assert copied[0] is graph
    assert matches == []


def test_canonicalize_graphs_does_not_mutate_input_graphs():
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Widget", type="Gadget", description="desc")],
        edges=[
            KGEdge(
                source_node_id="n1",
                target_node_id="widget",
                relationship_name="ref",
            )
        ],
    )
    original_node = graph.nodes[0].model_copy(deep=True)
    original_edge = graph.edges[0].model_copy(deep=True)

    copied, _ = canonicalize_graphs([graph], [_make_chunk()], _CanonicalizeStubResolver())

    assert graph.nodes[0] == original_node
    assert graph.edges[0] == original_edge
    assert copied[0].nodes[0].name == "widget_canonical"
    assert copied[0].nodes[0].type == "gadget"


def test_canonicalize_graphs_resolves_each_unique_name_once():
    resolver = _CanonicalizeStubResolver()
    call_count = {"n": 0}
    original = resolver.get_subgraph

    def counting_get_subgraph(node_name, node_type="individuals", directed=True):
        call_count["n"] += 1
        return original(node_name, node_type, directed)

    resolver.get_subgraph = counting_get_subgraph
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="a1", name="Widget", type="Gadget", description="one"),
            Node(id="a2", name="widget", type="Gadget", description="two"),
        ],
        edges=[],
    )

    _, matches = canonicalize_graphs([graph], [chunk], resolver)

    assert call_count["n"] == 2
    assert len(matches) == 2
    assert matches[0].category == "classes"
    assert matches[1].category == "individuals"
    assert matches[1].canonical_uri == "https://example.test/ontology#widget_canonical"


def test_extend_graph_preserves_canonical_and_injected_ontology_uris():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Widget", type="Gadget", description="desc")],
        edges=[],
    )
    _, matches = canonicalize_graphs([graph], [chunk], _CanonicalizeStubResolver())

    type_node = EntityType(
        id=EntityType.id_for("gadget"),
        name="gadget",
        description="gadget",
    )
    entity_node = Entity(
        id=Entity.id_for("widget_canonical"),
        name="widget_canonical",
        description="desc",
    )
    added_nodes_map = {
        f"{type_node.id}_type": type_node,
        f"{entity_node.id}_entity": entity_node,
    }

    injected_nodes, _ = extend_graph_with_ontology(matches, added_nodes_map, {})

    assert type_node.ontology_uri == "https://example.test/ontology#gadget"
    assert entity_node.ontology_uri == "https://example.test/ontology#widget_canonical"
    related = injected_nodes[f"{Entity.id_for('related')}_entity"]
    assert related.ontology_uri == "https://example.test/ontology#related"


def test_canonicalize_graphs_rewrites_edge_endpoints_by_name_and_id():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Widget", type="Gadget", description="note")],
        edges=[
            KGEdge(
                source_node_id="n1",
                target_node_id="widget",
                relationship_name="ref",
            )
        ],
    )

    copied, _ = canonicalize_graphs([graph], [chunk], _CanonicalizeStubResolver())
    canonical = "widget_canonical"

    assert copied[0].nodes[0].id == canonical
    assert copied[0].edges[0].source_node_id == canonical
    assert copied[0].edges[0].target_node_id == canonical


def test_canonicalize_graphs_name_collision_is_deterministic():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="id-one", name="Alpha", type="Other", description="first"),
            Node(id="id-two", name="alpha", type="Other", description="second"),
        ],
        edges=[],
    )

    copied, _ = canonicalize_graphs([graph], [chunk], _CanonicalizeStubResolver())

    assert copied[0].nodes[0].id == "alpha"
    assert copied[0].nodes[0].name == "alpha"
    assert copied[0].nodes[1].id == "alpha"
    assert copied[0].nodes[1].name == "alpha"


def test_canonicalize_graphs_id_rewrite_does_not_steal_name_identity():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="alpha", name="Beta", type="Other", description="by name"),
            Node(id="other", name="alpha", type="Other", description="by id"),
        ],
        edges=[],
    )

    copied, _ = canonicalize_graphs([graph], [chunk], _CanonicalizeStubResolver())

    assert copied[0].nodes[0].id == "alpha"
    assert copied[0].nodes[0].name == "Beta"
    assert copied[0].nodes[1].id == "alpha"
    assert copied[0].nodes[1].name == "alpha"


def test_canonicalize_graphs_mixed_endpoint_references_rewrite_consistently():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Widget", type="Gadget", description="desc")],
        edges=[
            KGEdge(source_node_id="n1", target_node_id="widget", relationship_name="by_id"),
            KGEdge(source_node_id="widget", target_node_id="Widget", relationship_name="by_name"),
        ],
    )

    copied, _ = canonicalize_graphs([graph], [chunk], _CanonicalizeStubResolver())
    canonical = "widget_canonical"

    assert copied[0].edges[0].source_node_id == canonical
    assert copied[0].edges[0].target_node_id == canonical
    assert copied[0].edges[1].source_node_id == canonical
    assert copied[0].edges[1].target_node_id == canonical


def test_canonicalize_graphs_leaves_unrelated_endpoints_untouched():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Widget", type="Gadget", description="desc")],
        edges=[
            KGEdge(
                source_node_id="unknown",
                target_node_id="missing",
                relationship_name="ref",
            )
        ],
    )

    copied, _ = canonicalize_graphs([graph], [chunk], _CanonicalizeStubResolver())

    assert copied[0].edges[0].source_node_id == "unknown"
    assert copied[0].edges[0].target_node_id == "missing"
