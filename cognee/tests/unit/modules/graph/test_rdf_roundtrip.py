"""Tests for RDF URI preservation (Phase 0), RDF export + SPARQL (Phase 1),
and RDF ingestion (Phase 2)."""

import importlib

import pytest
from rdflib import Graph, Literal, RDF, RDFS, URIRef

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.rdf.export import (
    DEFAULT_BASE_IRI,
    graph_data_to_rdf,
)
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.ontology.rdf_xml.rdf_ingest import (
    build_datapoints_from_rdf,
    build_graph_from_rdf,
    ingest_rdf,
)

MM = "http://example.org/mm#"

SAMPLE_TTL = f"""
@prefix ex: <{MM}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Machine a owl:Class ; rdfs:label "Machine" .
ex:CNCMachine a owl:Class ; rdfs:subClassOf ex:Machine ; rdfs:label "CNC Machine" .
ex:Plant a owl:Class .

ex:machine_42 a ex:CNCMachine ; rdfs:label "Machine 42" ; ex:locatedIn ex:plant_1 .
ex:plant_1 a ex:Plant ; rdfs:label "Plant 1" .
"""

COLLISION_TTL = """
@prefix ex1: <http://example.org/ns1#> .
@prefix ex2: <http://example.org/ns2#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

ex1:Thing a owl:Class .
ex1:a a ex1:Thing ; ex1:relatedTo ex1:b ; ex2:relatedTo ex1:b .
ex1:b a ex1:Thing .
"""


# --- Phase 0: the field exists and survives serialization ---------------------


def test_datapoint_has_ontology_uri_field_defaulting_none():
    dp = Entity(name="widget", description="a widget")
    assert dp.ontology_uri is None

    grounded = Entity(name="machine", description="m", ontology_uri=f"{MM}machine_42")
    assert grounded.ontology_uri == f"{MM}machine_42"
    # Round-trips through (de)serialization used by the storage layer.
    assert Entity.from_json(grounded.to_json()).ontology_uri == f"{MM}machine_42"


# --- Phase 2: RDF -> DataPoints, preserving IRIs (open-world) ------------------


def test_build_datapoints_from_rdf_preserves_uris_and_structure():
    graph = Graph()
    graph.parse(data=SAMPLE_TTL, format="turtle")

    ingest_graph = build_graph_from_rdf(graph)
    data_points = ingest_graph.data_points
    by_uri = {dp.ontology_uri: dp for dp in data_points}

    # 3 classes + 2 individuals
    types = [dp for dp in data_points if isinstance(dp, EntityType)]
    entities = [dp for dp in data_points if isinstance(dp, Entity)]
    assert len(types) == 3
    assert len(entities) == 2

    # Every node keeps its verbatim external IRI (nothing canonicalized).
    assert f"{MM}Machine" in by_uri
    assert f"{MM}CNCMachine" in by_uri
    assert f"{MM}machine_42" in by_uri

    # Labels are used for names when present.
    assert by_uri[f"{MM}machine_42"].name == "Machine 42"
    assert by_uri[f"{MM}CNCMachine"].name == "CNC Machine"

    # Identity is derived from the IRI, so it is stable/idempotent.
    assert by_uri[f"{MM}machine_42"].id == Entity.id_for(f"{MM}machine_42")

    # machine_42 --is_a--> CNCMachine
    machine = by_uri[f"{MM}machine_42"]
    assert machine.is_a is not None
    assert machine.is_a.ontology_uri == f"{MM}CNCMachine"

    # CNCMachine --is_a(subClassOf)--> Machine
    cnc = by_uri[f"{MM}CNCMachine"]
    is_a_targets = [tgt.ontology_uri for _edge, tgt in cnc.relations]
    assert f"{MM}Machine" in is_a_targets

    # object property assertion machine_42 --locatedIn--> plant_1 is explicit custom edge
    rel_names = {edge.relationship_type for edge, _tgt in machine.relations}
    assert "locatedIn" not in rel_names

    assert len(ingest_graph.custom_edges) == 1
    located_edge = ingest_graph.custom_edges[0]
    assert located_edge[0] == str(machine.id)
    assert located_edge[1] == str(by_uri[f"{MM}plant_1"].id)
    assert located_edge[2].startswith("rdf_locatedin_")
    assert located_edge[3]["predicate_uri"] == f"{MM}locatedIn"


def test_build_datapoints_from_rdf_keeps_node_only_compatibility():
    graph = Graph()
    graph.parse(data=SAMPLE_TTL, format="turtle")

    data_points = build_datapoints_from_rdf(graph)
    assert {dp.ontology_uri for dp in data_points} == {
        f"{MM}Machine",
        f"{MM}CNCMachine",
        f"{MM}Plant",
        f"{MM}machine_42",
        f"{MM}plant_1",
    }


# --- Phase 1: property-graph tuples -> RDF + SPARQL ---------------------------


def _sample_property_graph():
    """Mirror what GraphDBInterface.get_graph_data returns after Phase 0:
    props carry name / type / ontology_uri."""
    nodes = [
        (
            "id_machine",
            {"name": "Machine 42", "type": "Entity", "ontology_uri": f"{MM}machine_42"},
        ),
        ("id_plant", {"name": "Plant 1", "type": "Entity", "ontology_uri": f"{MM}plant_1"}),
        ("id_widget", {"name": "Widget", "type": "Entity"}),  # ungrounded -> minted IRI
        ("id_cnc", {"name": "CNC", "type": "EntityType", "ontology_uri": f"{MM}CNCMachine"}),
        (
            "id_machine_cls",
            {"name": "Machine", "type": "EntityType", "ontology_uri": f"{MM}Machine"},
        ),
    ]
    edges = [
        ("id_machine", "id_cnc", "is_a", {}),  # individual -> class => rdf:type
        ("id_cnc", "id_machine_cls", "is_a", {}),  # class -> class => rdfs:subClassOf
        ("id_machine", "id_plant", "locatedIn", {}),  # minted predicate
    ]
    return nodes, edges


def test_graph_data_to_rdf_uses_preserved_uris_and_rdf_semantics():
    nodes, edges = _sample_property_graph()
    g = graph_data_to_rdf(nodes, edges)

    # is_a resolves to rdf:type for individual->class and subClassOf for class->class
    assert (URIRef(f"{MM}machine_42"), RDF.type, URIRef(f"{MM}CNCMachine")) in g
    assert (URIRef(f"{MM}CNCMachine"), RDFS.subClassOf, URIRef(f"{MM}Machine")) in g

    # labels
    assert (URIRef(f"{MM}machine_42"), RDFS.label, Literal("Machine 42")) in g

    # minted predicate for a non-is_a relationship
    located = URIRef(f"{DEFAULT_BASE_IRI}prop/locatedIn")
    assert (URIRef(f"{MM}machine_42"), located, URIRef(f"{MM}plant_1")) in g

    # ungrounded node gets a minted IRI, not dropped
    minted = URIRef(f"{DEFAULT_BASE_IRI}node/id_widget")
    assert (minted, RDFS.label, Literal("Widget")) in g


def test_graph_data_to_rdf_uses_predicate_uri_when_present():
    nodes, edges = _sample_property_graph()
    edges.append(
        (
            "id_machine",
            "id_plant",
            "rdf_locatedin_123",
            {"predicate_uri": f"{MM}locatedIn", "edge_text": "locatedIn"},
        )
    )

    g = graph_data_to_rdf(nodes, edges)

    assert (URIRef(f"{MM}machine_42"), URIRef(f"{MM}locatedIn"), URIRef(f"{MM}plant_1")) in g


def test_sparql_query_over_memory_graph_rdf_view():
    nodes, edges = _sample_property_graph()
    g = graph_data_to_rdf(nodes, edges)

    rows = list(g.query(f"SELECT ?s WHERE {{ ?s a <{MM}CNCMachine> }}"))
    subjects = {str(row[0]) for row in rows}
    assert f"{MM}machine_42" in subjects

    # subclass relationship is queryable
    rows = list(g.query(f"SELECT ?c WHERE {{ <{MM}CNCMachine> rdfs:subClassOf ?c }}"))
    assert f"{MM}Machine" in {str(row[0]) for row in rows}


# --- Phases 2 -> 1 combined: ingest RDF, re-export, IRIs survive --------------


async def _graph_data_from_ingest_graph(ingest_graph):
    nodes = []
    model_edges = []
    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    for data_point in ingest_graph.data_points:
        extracted_nodes, extracted_edges = await get_graph_from_model(
            data_point,
            added_nodes=added_nodes,
            added_edges=added_edges,
            visited_properties=visited_properties,
        )
        nodes.extend(
            (
                str(node.id),
                {
                    "name": node.name,
                    "type": type(node).__name__,
                    "ontology_uri": node.ontology_uri,
                },
            )
            for node in extracted_nodes
        )
        model_edges.extend(extracted_edges)

    return nodes, model_edges + ingest_graph.custom_edges


@pytest.mark.asyncio
async def test_ingested_datapoints_reexport_preserves_external_iris():
    graph = Graph()
    graph.parse(data=SAMPLE_TTL, format="turtle")
    ingest_graph = build_graph_from_rdf(graph)

    nodes, edges = await _graph_data_from_ingest_graph(ingest_graph)
    g = graph_data_to_rdf(nodes, edges)

    # The external IRIs from the source RDF are present again after the trip
    # through cognee datapoints and back to RDF.
    assert (URIRef(f"{MM}machine_42"), RDF.type, URIRef(f"{MM}CNCMachine")) in g
    assert (URIRef(f"{MM}CNCMachine"), RDFS.subClassOf, URIRef(f"{MM}Machine")) in g
    assert (URIRef(f"{MM}machine_42"), URIRef(f"{MM}locatedIn"), URIRef(f"{MM}plant_1")) in g


@pytest.mark.asyncio
async def test_rdf_predicates_with_same_local_name_roundtrip_separately():
    graph = Graph()
    graph.parse(data=COLLISION_TTL, format="turtle")
    ingest_graph = build_graph_from_rdf(graph)

    assert len(ingest_graph.custom_edges) == 2
    relationship_names = {edge[2] for edge in ingest_graph.custom_edges}
    predicate_uris = {edge[3]["predicate_uri"] for edge in ingest_graph.custom_edges}

    assert len(relationship_names) == 2
    assert predicate_uris == {
        "http://example.org/ns1#relatedTo",
        "http://example.org/ns2#relatedTo",
    }

    nodes, edges = await _graph_data_from_ingest_graph(ingest_graph)
    g = graph_data_to_rdf(nodes, edges)

    assert (
        URIRef("http://example.org/ns1#a"),
        URIRef("http://example.org/ns1#relatedTo"),
        URIRef("http://example.org/ns1#b"),
    ) in g
    assert (
        URIRef("http://example.org/ns1#a"),
        URIRef("http://example.org/ns2#relatedTo"),
        URIRef("http://example.org/ns1#b"),
    ) in g


@pytest.mark.asyncio
async def test_ingest_rdf_passes_custom_edges_to_add_data_points(monkeypatch):
    graph = Graph()
    graph.parse(data=SAMPLE_TTL, format="turtle")
    captured = {}

    async def fake_add_data_points(data_points, custom_edges=None, ctx=None):
        captured["data_points"] = data_points
        captured["custom_edges"] = custom_edges
        captured["ctx"] = ctx
        return data_points

    add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")
    monkeypatch.setattr(add_data_points_module, "add_data_points", fake_add_data_points)

    result = await ingest_rdf(graph)
    expected = build_graph_from_rdf(graph)

    assert result == captured["data_points"]
    assert {dp.ontology_uri for dp in captured["data_points"]} == {
        dp.ontology_uri for dp in expected.data_points
    }
    assert captured["custom_edges"] == expected.custom_edges
