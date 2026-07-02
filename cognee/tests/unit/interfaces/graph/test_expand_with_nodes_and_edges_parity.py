"""Characterization tests for expand_with_nodes_and_edges ontology parity."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    _create_edge_key,
    expand_with_nodes_and_edges,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.get_default_ontology_resolver import get_default_ontology_resolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


class _DuplicateTwinResolver(BaseOntologyResolver):
    """Chunk 1 class match injects shared_p; chunk 2 extracts the same individual."""

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "team":
            root = AttachedOntologyNode("team", "classes")
            twin = AttachedOntologyNode("shared_p", "individuals")
            return [root, twin], [], root
        if node_type == "individuals" and node_name == "shared_p":
            root = AttachedOntologyNode("shared_p", "individuals")
            return [root], [], root
        return [], [], None


class _EdgeCollisionResolver(BaseOntologyResolver):
    """Ontology and extracted edges share the audi -> car is_a key."""

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "car":
            root = AttachedOntologyNode("car", "classes")
            return [root], [], root
        if node_type == "individuals" and node_name == "audi":
            root = AttachedOntologyNode("audi", "individuals")
            parent = AttachedOntologyNode("car", "classes")
            return [root, parent], [("audi", "is_a", "car")], root
        return [], [], None


class _PositiveOntologyEdgeResolver(BaseOntologyResolver):
    """Minimal resolver that emits one ontology is_a relation."""

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "car":
            root = AttachedOntologyNode("car", "classes")
            return [root], [], root
        if node_type == "individuals" and node_name == "audi":
            root = AttachedOntologyNode("audi", "individuals")
            parent = AttachedOntologyNode("car", "classes")
            return [root, parent], [("audi", "is_a", "car")], root
        return [], [], None


def _belongs_to_set_tuple(belongs_to_set) -> tuple[str, ...]:
    if not belongs_to_set:
        return ()
    return tuple(sorted(str(getattr(item, "id", item)) for item in belongs_to_set))


def _serialize_node(node) -> tuple:
    is_a = getattr(node, "is_a", None)
    return (
        node.__class__.__name__,
        str(node.id),
        node.name,
        node.type,
        getattr(node, "description", None),
        str(is_a.id) if is_a is not None else None,
        is_a.name if is_a is not None else None,
        node.ontology_valid,
        node.importance_weight,
        _belongs_to_set_tuple(getattr(node, "belongs_to_set", None)),
    )


def _serialize_relations(entity_nodes) -> tuple[tuple, ...]:
    relations = []
    for node in entity_nodes:
        for edge_obj, target in node.relations:
            relations.append(
                (
                    str(node.id),
                    edge_obj.relationship_type,
                    str(target.id),
                    edge_obj.edge_text,
                )
            )
    return tuple(sorted(relations, key=lambda row: (row[0], row[1], row[2], row[3] or "")))


def _serialize_contains(chunks) -> tuple[tuple, ...]:
    contains = []
    for chunk in chunks:
        if not chunk.contains:
            continue
        for edge_obj, entity in chunk.contains:
            contains.append(
                (
                    edge_obj.relationship_type,
                    edge_obj.edge_text,
                    str(entity.id),
                )
            )
    return tuple(sorted(contains, key=lambda row: (row[0], row[1] or "", row[2])))


def serialize_output(chunks, entity_nodes) -> dict[str, Any]:
    """Turn expand output into a stable, diff-friendly structure."""
    nodes = tuple(sorted((_serialize_node(node) for node in entity_nodes), key=lambda row: row[1]))
    return {
        "nodes": nodes,
        "relations": _serialize_relations(entity_nodes),
        "contains": _serialize_contains(chunks),
    }


def _relation_triples(entity_nodes) -> set[tuple[str, str, str]]:
    """Return (source_name, relationship_type, target_name) for attached relations."""
    triples = set()
    for node in entity_nodes:
        for edge_obj, target in node.relations:
            triples.add((node.name, edge_obj.relationship_type, target.name))
    return triples


def run_duplicate_twin_scenario():
    """Chunk 1 injects shared_p; chunk 2 extracts the same canonical individual."""
    chunk1 = _make_chunk(importance_weight=0.9)
    chunk2 = _make_chunk(importance_weight=0.1)
    graph1 = _make_graph(
        [Node(id="anchor", name="Anchor", type="Team", description="chunk1 anchor")],
        [],
    )
    graph2 = _make_graph(
        [Node(id="ext", name="shared_p", type="Other", description="chunk2 extracted")],
        [],
    )
    return expand_with_nodes_and_edges([chunk1, chunk2], [graph1, graph2], _DuplicateTwinResolver())


def run_edge_collision_scenario():
    """Extracted and ontology edges collide on audi -> car is_a."""
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="e1", name="Audi", type="Car", description="audi entity")],
        [
            KGEdge(
                source_node_id="audi",
                target_node_id="car",
                relationship_name="is_a",
                description="extracted is_a wins later",
            )
        ],
    )
    return expand_with_nodes_and_edges([chunk], [graph], _EdgeCollisionResolver())


def _make_chunk(importance_weight=0.5, belongs_to_set=None):
    chunk = MagicMock()
    chunk.contains = None
    chunk.belongs_to_set = belongs_to_set if belongs_to_set is not None else []
    chunk.importance_weight = importance_weight
    return chunk


def _make_graph(nodes, edges):
    return KnowledgeGraph(nodes=nodes, edges=edges)


def _build_scenario_b_ontology() -> RDFLibOntologyResolver:
    """Small in-memory ontology covering exact, fuzzy, hierarchy, and property edges."""
    ns = Namespace("http://example.org/parity#")
    graph = Graph()
    graph.add((ns.Person, RDF.type, OWL.Class))
    graph.add((ns.Company, RDF.type, OWL.Class))
    graph.add((ns.Vehicle, RDF.type, OWL.Class))
    graph.add((ns.Car, RDF.type, OWL.Class))
    graph.add((ns.Vehicle, RDFS.subClassOf, OWL.Thing))
    graph.add((ns.Car, RDFS.subClassOf, ns.Vehicle))
    graph.add((ns.Alice, RDF.type, ns.Person))
    graph.add((ns.Audi, RDF.type, ns.Car))
    graph.add((ns.porsche_911, RDF.type, ns.Car))
    graph.add((ns.VW, RDF.type, ns.Company))
    graph.add((ns.owns, RDF.type, OWL.ObjectProperty))
    graph.add((ns.VW, ns.owns, ns.Audi))

    turtle = graph.serialize(format="turtle")
    resolver = RDFLibOntologyResolver(ontology_file=io.StringIO(turtle))
    return resolver


def run_scenario_a(resolver=None):
    """No-ontology fixture: dedup, blank description, id!=name, duplicate edges."""
    if resolver is None:
        resolver = get_default_ontology_resolver()

    chunk1 = _make_chunk(importance_weight=0.7)
    chunk2 = _make_chunk(importance_weight=0.3)
    graph1 = _make_graph(
        [
            Node(id="n1", name="Alice", type="Person", description="Alice founded Acme."),
            Node(id="uuid-charlie", name="Charlie", type="Person", description="Charlie note."),
            Node(id="n-blank", name="Bob", type="Person", description="   "),
        ],
        [
            KGEdge(source_node_id="n1", target_node_id="n-blank", relationship_name="knows"),
            KGEdge(source_node_id="n1", target_node_id="n-blank", relationship_name="knows"),
            KGEdge(
                source_node_id="uuid-charlie",
                target_node_id="n1",
                relationship_name="met",
                description="Charlie met Alice.",
            ),
        ],
    )
    graph2 = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="Alice lives in Paris.")],
        [KGEdge(source_node_id="n1", target_node_id="n-blank", relationship_name="knows")],
    )
    return expand_with_nodes_and_edges([chunk1, chunk2], [graph1, graph2], resolver)


def run_scenario_b(resolver=None):
    """Matching-ontology fixture exercising canonicalization, injection, and collisions."""
    if resolver is None:
        resolver = _build_scenario_b_ontology()

    marker = "set-marker-1"
    chunk_main = _make_chunk(importance_weight=0.99, belongs_to_set=[marker])
    chunk_extra = _make_chunk(importance_weight=0.4)

    graph_main = _make_graph(
        [
            Node(id="e-audi", name="Audi", type="Car", description="Audi from chunk."),
            Node(id="porsche_911s", name="porsche_911s", type="Car", description="Porsche note."),
            Node(
                id="ghost-1", name="ghost_entity", type="Phantom", description="No ontology match."
            ),
            Node(id="collide-1", name="audi", type="Car", description="First collision desc."),
            Node(id="collide-2", name="Audi", type="Car", description="Second collision desc."),
        ],
        [
            KGEdge(
                source_node_id="porsche_911s",
                target_node_id="audi",
                relationship_name="raced_against",
                description="Porsche raced Audi.",
            ),
        ],
    )
    graph_extra = _make_graph(
        [
            Node(
                id="n-alice", name="Alice", type="Person", description="Exact type and entity path."
            ),
        ],
        [],
    )
    return expand_with_nodes_and_edges(
        [chunk_main, chunk_extra], [graph_main, graph_extra], resolver
    )


# Characterization of current dev behavior — generated via temp/graph_model_unification/generate_parity_goldens.py
GOLDEN_A = {
    "contains": (
        ("contains", None, "8b13b063-8456-5ac8-b360-254105b73090"),
        (
            "contains",
            "Document chunk mentions alice: Alice founded Acme.",
            "6e9db765-ff0a-5c4a-a6a7-10d01875dd0f",
        ),
        (
            "contains",
            "Document chunk mentions alice: Alice lives in Paris.",
            "6e9db765-ff0a-5c4a-a6a7-10d01875dd0f",
        ),
        (
            "contains",
            "Document chunk mentions charlie: Charlie note.",
            "2406b1cf-5790-5dd2-861c-17ce17ce0ebe",
        ),
    ),
    "nodes": (
        (
            "Entity",
            "2406b1cf-5790-5dd2-861c-17ce17ce0ebe",
            "charlie",
            "Entity",
            "Charlie note.",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            False,
            0.7,
            (),
        ),
        (
            "Entity",
            "6e9db765-ff0a-5c4a-a6a7-10d01875dd0f",
            "alice",
            "Entity",
            "Alice founded Acme.",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            False,
            0.7,
            (),
        ),
        (
            "Entity",
            "8b13b063-8456-5ac8-b360-254105b73090",
            "bob",
            "Entity",
            "   ",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            False,
            0.7,
            (),
        ),
        (
            "EntityType",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            "EntityType",
            "person",
            None,
            None,
            False,
            0.7,
            (),
        ),
    ),
    "relations": (
        (
            "2406b1cf-5790-5dd2-861c-17ce17ce0ebe",
            "met",
            "6e9db765-ff0a-5c4a-a6a7-10d01875dd0f",
            "Charlie met Alice.",
        ),
        (
            "6e9db765-ff0a-5c4a-a6a7-10d01875dd0f",
            "knows",
            "8b13b063-8456-5ac8-b360-254105b73090",
            None,
        ),
    ),
}

GOLDEN_B = {
    "contains": (
        (
            "contains",
            "Document chunk mentions alice: Exact type and entity path.",
            "afc75e41-4df3-5db9-907e-dcf55750efec",
        ),
        (
            "contains",
            "Document chunk mentions audi: Audi from chunk.",
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
        ),
        (
            "contains",
            "Document chunk mentions audi: First collision desc.",
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
        ),
        (
            "contains",
            "Document chunk mentions audi: Second collision desc.",
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
        ),
        (
            "contains",
            "Document chunk mentions ghost_entity: No ontology match.",
            "4ee7f575-8992-54fa-9e8d-41691e27dc67",
        ),
        (
            "contains",
            "Document chunk mentions porsche_911: Porsche note.",
            "013e367e-5401-5a01-8505-f13ad52b3933",
        ),
    ),
    "nodes": (
        (
            "Entity",
            "013e367e-5401-5a01-8505-f13ad52b3933",
            "porsche_911",
            "Entity",
            "Porsche note.",
            "e4bbd678-157f-5349-ad0d-b822c3021210",
            "car",
            True,
            0.99,
            ("set-marker-1",),
        ),
        (
            "EntityType",
            "0f5ec78d-c353-540c-b8bd-bce84eac0af4",
            "vehicle",
            "EntityType",
            "vehicle",
            None,
            None,
            True,
            0.99,
            (),
        ),
        (
            "Entity",
            "4ee7f575-8992-54fa-9e8d-41691e27dc67",
            "ghost_entity",
            "Entity",
            "No ontology match.",
            "6fe754cc-42b8-5303-be5e-d976ccd73834",
            "phantom",
            False,
            0.99,
            ("set-marker-1",),
        ),
        (
            "Entity",
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
            "audi",
            "Entity",
            "Second collision desc.",
            "e4bbd678-157f-5349-ad0d-b822c3021210",
            "car",
            True,
            0.99,
            ("set-marker-1",),
        ),
        (
            "EntityType",
            "6fe754cc-42b8-5303-be5e-d976ccd73834",
            "phantom",
            "EntityType",
            "phantom",
            None,
            None,
            False,
            0.99,
            (),
        ),
        (
            "EntityType",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            "EntityType",
            "person",
            None,
            None,
            True,
            0.4,
            (),
        ),
        (
            "Entity",
            "afc75e41-4df3-5db9-907e-dcf55750efec",
            "alice",
            "Entity",
            "Exact type and entity path.",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            True,
            0.4,
            (),
        ),
        (
            "EntityType",
            "e4bbd678-157f-5349-ad0d-b822c3021210",
            "car",
            "EntityType",
            "car",
            None,
            None,
            True,
            0.99,
            (),
        ),
    ),
    "relations": (
        (
            "013e367e-5401-5a01-8505-f13ad52b3933",
            "raced_against",
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
            "Porsche raced Audi.",
        ),
    ),
}


def test_scenario_a_no_ontology_golden():
    chunks, entity_nodes = run_scenario_a()
    output = serialize_output(chunks, entity_nodes)
    assert output == GOLDEN_A
    assert len({row[1] for row in output["nodes"]}) == len(output["nodes"])


@pytest.mark.parametrize(
    "resolver",
    [get_default_ontology_resolver(), None],
    ids=["empty_resolver", "skip_none"],
)
def test_scenario_a_skip_equals_empty_resolver(resolver):
    """Skip path must match today's empty-resolver graph."""
    chunks, entity_nodes = run_scenario_a(resolver)
    assert serialize_output(chunks, entity_nodes) == GOLDEN_A


def test_scenario_b_matching_ontology_golden():
    chunks, entity_nodes = run_scenario_b()
    output = serialize_output(chunks, entity_nodes)
    assert output == GOLDEN_B
    assert len({row[1] for row in output["nodes"]}) == len(output["nodes"])
    # Deliberate characterization: second collision node wins under the canonical audi key.
    audi_nodes = [row for row in output["nodes"] if row[2] == "audi" and row[0] == "Entity"]
    assert len(audi_nodes) == 1
    assert audi_nodes[0][4] == "Second collision desc."


def test_positive_ontology_edge_pin():
    """Prove ontology edge injection attaches at least one ontology-derived relation."""
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="e1", name="Audi", type="Car", description="audi entity")],
        [],
    )
    _, entity_nodes = expand_with_nodes_and_edges([chunk], [graph], _PositiveOntologyEdgeResolver())
    assert ("audi", "is_a", "car") in _relation_triples(entity_nodes)


def test_duplicate_twin_pin_two_nodes_same_id():
    """Phase 1 should collapse injected and extracted twins sharing one canonical id."""
    _, entity_nodes = run_duplicate_twin_scenario()
    shared_id = str(Entity.id_for("shared_p"))
    shared_nodes = [node for node in entity_nodes if str(node.id) == shared_id]
    assert len(shared_nodes) == 2


def test_edge_collision_pin_ontology_edge_wins_today():
    """Phase 1 should let the extracted edge win on a shared key (delta 3)."""
    _, entity_nodes = run_edge_collision_scenario()
    is_a_rels = [
        (edge.edge_text,)
        for node in entity_nodes
        for edge, target in node.relations
        if edge.relationship_type == "is_a" and node.name == "audi" and target.name == "car"
    ]
    assert len(is_a_rels) == 1
    assert is_a_rels[0] == (None,)


def test_scenario_c_dedup_before_matching_call_count():
    resolver = _build_scenario_b_ontology()
    call_count = {"n": 0}
    original = resolver.get_subgraph

    def counting_get_subgraph(node_name, node_type="individuals", directed=True):
        call_count["n"] += 1
        return original(node_name, node_type, directed)

    resolver.get_subgraph = counting_get_subgraph
    run_scenario_b(resolver)
    # 3 types (Car, Phantom, Person) + 6 entities
    assert call_count["n"] == 9


def test_scenario_d_existing_edges_map_filters_relations():
    resolver = _build_scenario_b_ontology()
    extracted_edge_key = _create_edge_key(
        str(Entity.id_for("porsche_911")),
        str(Entity.id_for("audi")),
        generate_edge_name("raced_against"),
    )
    ontology_edge_key = _create_edge_key(
        str(Entity.id_for("audi")),
        str(EntityType.id_for("car")),
        generate_edge_name("is_a"),
    )
    seeded = {extracted_edge_key: True, ontology_edge_key: True}
    chunk_main = _make_chunk(importance_weight=0.99, belongs_to_set=["set-marker-1"])
    chunk_extra = _make_chunk(importance_weight=0.4)
    graph_main = _make_graph(
        [
            Node(id="e-audi", name="Audi", type="Car", description="Audi from chunk."),
            Node(id="porsche_911s", name="porsche_911s", type="Car", description="Porsche note."),
            Node(
                id="ghost-1", name="ghost_entity", type="Phantom", description="No ontology match."
            ),
            Node(id="collide-1", name="audi", type="Car", description="First collision desc."),
            Node(id="collide-2", name="Audi", type="Car", description="Second collision desc."),
        ],
        [
            KGEdge(
                source_node_id="porsche_911s",
                target_node_id="audi",
                relationship_name="raced_against",
            ),
        ],
    )
    graph_extra = _make_graph(
        [
            Node(
                id="n-alice", name="Alice", type="Person", description="Exact type and entity path."
            )
        ],
        [],
    )
    _, filtered_nodes = expand_with_nodes_and_edges(
        [chunk_main, chunk_extra], [graph_main, graph_extra], resolver, seeded
    )
    filtered = serialize_output([chunk_main, chunk_extra], filtered_nodes)
    filtered_rels = {rel for rel in filtered["relations"]}
    assert not any(rel[1] == "raced_against" for rel in filtered_rels)
    assert not any(rel[1] == "is_a" for rel in filtered_rels)
