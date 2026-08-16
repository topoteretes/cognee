from unittest.mock import MagicMock

import pytest

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    add_ontology_data_points_and_edges,
    canonicalize_extracted_graphs,
    construct_data_points_and_edges_with_ontology,
)
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


class _StubResolver(BaseOntologyResolver):
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
            return [root, related], [(root.name, "related_to", related.name)], root
        return [], [], None


def _make_chunk():
    chunk = MagicMock()
    chunk.importance_weight = 0.5
    chunk.belongs_to_set = []
    return chunk


def _make_graph():
    return KnowledgeGraph(
        nodes=[Node(id="local-widget", name="Widget", type="Gadget", description="desc")],
        edges=[],
    )


def test_canonicalize_extracted_graphs_looks_up_each_name_once_and_updates_in_place():
    resolver = _StubResolver()
    resolver.get_subgraph = MagicMock(wraps=resolver.get_subgraph)
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="local-widget-1", name="Widget", type="Gadget", description="first"),
            Node(id="local-widget-2", name="widget", type="Gadget", description="second"),
        ],
        edges=[],
    )
    extracted_graphs = [graph]

    canonicalized_graphs, ontology_match_lookup = canonicalize_extracted_graphs(
        [chunk],
        extracted_graphs,
        resolver,
    )

    assert canonicalized_graphs is extracted_graphs
    assert resolver.get_subgraph.call_count == 2
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "local-widget-1"
    assert graph.nodes[0].name == "widget_canonical"
    assert graph.nodes[0].type == "gadget"

    matches_by_category = {
        match.node_category: match for match in ontology_match_lookup.values() if match is not None
    }
    entity_ontology_match = matches_by_category["individuals"]
    entity_type_ontology_match = matches_by_category["classes"]
    assert entity_ontology_match.canonical_name == "widget_canonical"
    assert entity_ontology_match.canonical_uri == ("https://example.test/ontology#widget_canonical")
    assert entity_ontology_match.first_source_chunk is chunk
    assert entity_type_ontology_match.canonical_name == "gadget"


def test_canonicalize_extracted_graphs_rewrites_edges_from_collapsed_nodes():
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="widget-1", name="Widget", type="Gadget", description="first"),
            Node(id="widget-2", name="widget", type="Gadget", description="second"),
        ],
        edges=[
            KGEdge(
                source_node_id="widget-2",
                target_node_id="widget-1",
                relationship_name="same_as",
            )
        ],
    )

    canonicalize_extracted_graphs([chunk], [graph], _StubResolver())

    assert [node.id for node in graph.nodes] == ["widget-1"]
    assert graph.edges[0].source_node_id == "widget-1"
    assert graph.edges[0].target_node_id == "widget-1"


def test_canonicalize_extracted_graphs_rejects_duplicate_local_node_ids():
    graph = KnowledgeGraph(
        nodes=[
            Node(id="duplicate", name="Widget", type="Gadget", description="first"),
            Node(id="duplicate", name="widget", type="Gadget", description="second"),
        ],
        edges=[],
    )

    with pytest.raises(ValueError, match="Duplicate node id in extracted graph: duplicate"):
        canonicalize_extracted_graphs([_make_chunk()], [graph], _StubResolver())


def test_add_ontology_data_points_and_edges_adds_only_missing_items():
    chunk = _make_chunk()
    _, ontology_match_lookup = canonicalize_extracted_graphs(
        [chunk],
        [_make_graph()],
        _StubResolver(),
    )
    gadget = EntityType(
        id=EntityType.id_for("gadget"),
        name="gadget",
        description="gadget",
    )
    widget = Entity(
        id=Entity.id_for("widget_canonical"),
        name="widget_canonical",
        description="desc",
    )
    data_points_by_id = {
        str(gadget.id): gadget,
        str(widget.id): widget,
    }
    edges_by_identity = {}

    add_ontology_data_points_and_edges(
        ontology_match_lookup,
        data_points_by_id,
        edges_by_identity,
    )

    assert len([node for node in data_points_by_id.values() if node.name == "gadget"]) == 1
    assert (
        len([node for node in data_points_by_id.values() if node.name == "widget_canonical"]) == 1
    )
    assert gadget.ontology_valid is True
    assert gadget.ontology_uri == "https://example.test/ontology#gadget"
    assert widget.ontology_valid is True
    assert widget.ontology_uri == "https://example.test/ontology#widget_canonical"
    related = data_points_by_id[str(Entity.id_for("related"))]
    assert related.ontology_uri == "https://example.test/ontology#related"
    assert len(edges_by_identity) == 1
    assert next(iter(edges_by_identity.values())).relationship_type == "related_to"


def test_construct_data_points_and_edges_with_ontology_uses_the_pure_constructor():
    chunk = _make_chunk()
    chunk.contains = None
    graph = _make_graph()

    data_points_by_id, _ = construct_data_points_and_edges_with_ontology(
        [chunk],
        [graph],
        _StubResolver(),
    )

    widget = data_points_by_id[str(Entity.id_for("widget_canonical"))]
    gadget = data_points_by_id[str(EntityType.id_for("gadget"))]
    assert widget.name == "widget_canonical"
    assert widget.is_a is gadget
    assert widget.ontology_valid is True
    assert gadget.ontology_valid is True
