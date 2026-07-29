"""Characterization tests for extracted graph construction and ontology enrichment."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    attach_new_edges_to_data_points,
    construct_data_points_and_edges,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.get_default_ontology_resolver import get_default_ontology_resolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    construct_data_points_and_edges_with_ontology,
)
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
    """Ontology and extracted edges share the audi -> bob knows key."""

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "person":
            root = AttachedOntologyNode("person", "classes")
            return [root], [], root
        if node_type == "individuals" and node_name == "audi":
            root = AttachedOntologyNode("audi", "individuals")
            bob = AttachedOntologyNode("bob", "individuals")
            return [root, bob], [("audi", "knows", "bob")], root
        if node_type == "individuals" and node_name == "bob":
            root = AttachedOntologyNode("bob", "individuals")
            return [root], [], root
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


def _serialize_relations(data_points) -> tuple[tuple, ...]:
    relations = []
    for node in data_points:
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


def serialize_output(chunks, data_points) -> dict[str, Any]:
    """Turn constructed graph data into a stable, diff-friendly structure."""
    nodes = tuple(sorted((_serialize_node(node) for node in data_points), key=lambda row: row[1]))
    return {
        "nodes": nodes,
        "relations": _serialize_relations(data_points),
        "contains": _serialize_contains(chunks),
    }


def _relation_triples(data_points) -> set[tuple[str, str, str]]:
    """Return (source_name, relationship_type, target_name) for attached relations."""
    triples = set()
    for node in data_points:
        for edge_obj, target in node.relations:
            triples.add((node.name, edge_obj.relationship_type, target.name))
    return triples


def _construct_test_output(
    data_chunks,
    extracted_graphs,
    ontology_resolver=None,
    existing_edge_identities=None,
):
    if ontology_resolver is None:
        data_points_by_id, edges_by_identity = construct_data_points_and_edges(
            data_chunks,
            extracted_graphs,
        )
    else:
        data_points_by_id, edges_by_identity = construct_data_points_and_edges_with_ontology(
            data_chunks,
            extracted_graphs,
            ontology_resolver,
        )
    attach_new_edges_to_data_points(
        data_points_by_id,
        edges_by_identity,
        existing_edge_identities or set(),
    )
    return data_chunks, list(data_points_by_id.values())


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
    return _construct_test_output([chunk1, chunk2], [graph1, graph2], _DuplicateTwinResolver())


def run_edge_collision_scenario():
    """Extracted and ontology edges collide on audi -> bob knows."""
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="e-audi", name="Audi", type="Person", description="audi entity"),
            Node(id="e-bob", name="Bob", type="Person", description="bob entity"),
        ],
        [
            KGEdge(
                source_node_id="e-audi",
                target_node_id="e-bob",
                relationship_name="knows",
                description="extracted knows wins later",
            )
        ],
    )
    return _construct_test_output([chunk], [graph], _EdgeCollisionResolver())


def _make_chunk(importance_weight=0.5, belongs_to_set=None):
    chunk = MagicMock()
    chunk.contains = None
    chunk.belongs_to_set = belongs_to_set if belongs_to_set is not None else []
    chunk.importance_weight = importance_weight
    return chunk


def _make_graph(nodes, edges):
    return KnowledgeGraph(nodes=nodes, edges=edges)


def _build_matching_ontology_resolver() -> RDFLibOntologyResolver:
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


_USE_EMPTY_ONTOLOGY_RESOLVER = object()


def run_no_ontology_scenario(resolver=_USE_EMPTY_ONTOLOGY_RESOLVER):
    """No-ontology fixture: dedup, blank description, id!=name, duplicate edges."""
    if resolver is _USE_EMPTY_ONTOLOGY_RESOLVER:
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
    return _construct_test_output([chunk1, chunk2], [graph1, graph2], resolver)


def run_matching_ontology_scenario(resolver=None):
    """Matching-ontology fixture exercising canonicalization, injection, and collisions."""
    if resolver is None:
        resolver = _build_matching_ontology_resolver()

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
                target_node_id="e-audi",
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
    return _construct_test_output([chunk_main, chunk_extra], [graph_main, graph_extra], resolver)


# Expected output after name-based entity ID calculation and graph construction.
EXPECTED_OUTPUT_WITHOUT_ONTOLOGY = {
    "contains": (
        ("contains", None, "6dd5cf03-5ad1-5e1a-b215-633dfd34abae"),
        (
            "contains",
            "Document chunk mentions alice: Alice founded Acme.",
            "afc75e41-4df3-5db9-907e-dcf55750efec",
        ),
        (
            "contains",
            "Document chunk mentions alice: Alice lives in Paris.",
            "afc75e41-4df3-5db9-907e-dcf55750efec",
        ),
        (
            "contains",
            "Document chunk mentions charlie: Charlie note.",
            "9503b475-c153-5b9b-9122-8e3ef0042c11",
        ),
    ),
    "nodes": (
        (
            "Entity",
            "6dd5cf03-5ad1-5e1a-b215-633dfd34abae",
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
            "Entity",
            "9503b475-c153-5b9b-9122-8e3ef0042c11",
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
        (
            "Entity",
            "afc75e41-4df3-5db9-907e-dcf55750efec",
            "alice",
            "Entity",
            "Alice founded Acme.",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            "person",
            False,
            0.7,
            (),
        ),
    ),
    "relations": (
        (
            "9503b475-c153-5b9b-9122-8e3ef0042c11",
            "met",
            "afc75e41-4df3-5db9-907e-dcf55750efec",
            "Charlie met Alice.",
        ),
        (
            "afc75e41-4df3-5db9-907e-dcf55750efec",
            "knows",
            "6dd5cf03-5ad1-5e1a-b215-633dfd34abae",
            None,
        ),
    ),
}

# Expected output after ontology canonicalization and enrichment.
EXPECTED_OUTPUT_WITH_ONTOLOGY = {
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
            "Document chunk mentions ghost_entity: No ontology match.",
            "5962d98a-e7b6-5854-8325-7ad7f15f72be",
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
            "5962d98a-e7b6-5854-8325-7ad7f15f72be",
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
            "Audi from chunk.",
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
            "is_a",
            "e4bbd678-157f-5349-ad0d-b822c3021210",
            None,
        ),
        (
            "013e367e-5401-5a01-8505-f13ad52b3933",
            "raced_against",
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
            "Porsche raced Audi.",
        ),
        (
            "5b4598fe-f589-57a1-88a8-d72c48f4915b",
            "is_a",
            "e4bbd678-157f-5349-ad0d-b822c3021210",
            None,
        ),
        (
            "afc75e41-4df3-5db9-907e-dcf55750efec",
            "is_a",
            "99a82eec-bb35-5bb5-9ed2-e73fc7a756de",
            None,
        ),
        (
            "e4bbd678-157f-5349-ad0d-b822c3021210",
            "is_a",
            "0f5ec78d-c353-540c-b8bd-bce84eac0af4",
            None,
        ),
    ),
}


def test_no_ontology_output():
    chunks, data_points = run_no_ontology_scenario()
    output = serialize_output(chunks, data_points)
    assert output == EXPECTED_OUTPUT_WITHOUT_ONTOLOGY
    assert len({row[1] for row in output["nodes"]}) == len(output["nodes"])


@pytest.mark.parametrize(
    "resolver",
    [get_default_ontology_resolver(), None],
    ids=["empty_resolver", "skip_none"],
)
def test_no_ontology_paths_have_the_same_output(resolver):
    chunks, data_points = run_no_ontology_scenario(resolver)
    assert serialize_output(chunks, data_points) == EXPECTED_OUTPUT_WITHOUT_ONTOLOGY


def test_matching_ontology_output():
    chunks, data_points = run_matching_ontology_scenario()
    output = serialize_output(chunks, data_points)
    assert output == EXPECTED_OUTPUT_WITH_ONTOLOGY
    assert len({row[1] for row in output["nodes"]}) == len(output["nodes"])
    # The first extracted node matched to canonical Audi survives duplicate collapse.
    audi_nodes = [row for row in output["nodes"] if row[2] == "audi" and row[0] == "Entity"]
    assert len(audi_nodes) == 1
    assert audi_nodes[0][4] == "Audi from chunk."


def test_positive_ontology_edge_pin():
    """Prove production resolver edges attach despite URI casing differences."""
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="e1", name="Audi", type="Car", description="audi entity")],
        [],
    )
    _, data_points = _construct_test_output(
        [chunk],
        [graph],
        _build_matching_ontology_resolver(),
    )
    assert ("audi", "is_a", "car") in _relation_triples(data_points)


def test_duplicate_twin_pin_single_node_after_post_pass_injection():
    """Post-pass injection dedupes against the complete converted node set."""
    _, data_points = run_duplicate_twin_scenario()
    shared_id = str(Entity.id_for("shared_p"))
    shared_nodes = [node for node in data_points if str(node.id) == shared_id]
    assert len(shared_nodes) == 1


def test_duplicate_twin_order_invariant():
    """Duplicate-twin output is stable when chunk order is permuted."""
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
    forward_chunks, forward_data_points = _construct_test_output(
        [chunk1, chunk2], [graph1, graph2], _DuplicateTwinResolver()
    )
    reverse_chunks, reverse_data_points = _construct_test_output(
        [chunk2, chunk1], [graph2, graph1], _DuplicateTwinResolver()
    )

    forward_out = serialize_output(forward_chunks, forward_data_points)
    reverse_out = serialize_output(reverse_chunks, reverse_data_points)
    assert forward_out["nodes"] == reverse_out["nodes"]
    assert forward_out["relations"] == reverse_out["relations"]


def test_edge_collision_pin_extracted_edge_wins():
    """Extracted edges are emitted before ontology edges on a shared key (delta 3)."""
    _, data_points = run_edge_collision_scenario()
    knows_rels = [
        edge.edge_text
        for node in data_points
        for edge, target in node.relations
        if edge.relationship_type == "knows" and node.name == "audi" and target.name == "bob"
    ]
    assert knows_rels == ["extracted knows wins later"]


def test_each_distinct_name_is_matched_once():
    resolver = _build_matching_ontology_resolver()
    call_count = {"n": 0}
    original = resolver.get_subgraph

    def counting_get_subgraph(node_name, node_type="individuals", directed=True):
        call_count["n"] += 1
        return original(node_name, node_type, directed)

    resolver.get_subgraph = counting_get_subgraph
    run_matching_ontology_scenario(resolver)
    assert call_count["n"] == 7


def test_existing_edges_are_not_attached_to_data_points():
    resolver = _build_matching_ontology_resolver()
    extracted_edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("porsche_911")),
        target_id=str(Entity.id_for("audi")),
        relationship_name=generate_edge_name("raced_against"),
    )
    ontology_edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("audi")),
        target_id=str(EntityType.id_for("car")),
        relationship_name=generate_edge_name("is_a"),
    )
    existing_edge_identities = {extracted_edge_identity, ontology_edge_identity}
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
                target_node_id="e-audi",
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
    _, filtered_data_points = _construct_test_output(
        [chunk_main, chunk_extra],
        [graph_main, graph_extra],
        resolver,
        existing_edge_identities,
    )
    filtered = serialize_output([chunk_main, chunk_extra], filtered_data_points)
    attached_edge_identities = {
        EdgeIdentity(
            source_id=source_id,
            target_id=target_id,
            relationship_name=relationship_name,
        )
        for source_id, relationship_name, target_id, _edge_text in filtered["relations"]
    }
    assert extracted_edge_identity not in attached_edge_identities
    assert ontology_edge_identity not in attached_edge_identities
