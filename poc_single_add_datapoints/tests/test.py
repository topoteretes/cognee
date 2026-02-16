import pytest
from types import SimpleNamespace
from uuid import uuid5, NAMESPACE_DNS
from unittest.mock import AsyncMock, Mock
from io import StringIO
import importlib

from cognee.modules.data.processing.document_types import Document
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.infrastructure.engine import Edge as EngineEdge
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.shared.data_models import KnowledgeGraph, Node, Edge

import poc_single_add_datapoints.poc_extract_graph_from_data as poc_mod
from poc_single_add_datapoints.poc_expand_with_nodes_and_edges import (
    poc_expand_with_nodes_and_edges,
)
from cognee.modules.graph.utils.expand_with_nodes_and_edges import expand_with_nodes_and_edges

core_module_path = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")


def _make_chunk():
    doc = Document(
        name="doc",
        raw_data_location="tmp/doc.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunk = DocumentChunk(
        text="Qubits use superposition. Qubits can have either values 0, 1 or be in superposition.",
        chunk_size=5,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=doc,
    )
    # Seed contains with the same nodes/edges as _kg
    kg = _kg()
    type_nodes = {}
    entity_nodes = {}

    for node in kg.nodes:
        type_name = node.type
        if type_name not in type_nodes:
            type_nodes[type_name] = EntityType(
                id=generate_node_id(node.description),
                name=type_name,
                type=type_name,
                description=type_name,
            )

        entity_nodes[node.id] = Entity(
            id=generate_node_id(node.description),
            name=node.name,
            description=node.description,
            is_a=type_nodes[type_name],
        )

    chunk.contains = [
        (EngineEdge(relationship_type="contains"), entity) for entity in entity_nodes.values()
    ]
    return chunk


def _kg():
    node = Node(id="n1", name="Qubit", type="Concept", description="q")
    node1 = Node(id="n2", name="1", type="Value", description="1")
    node2 = Node(id="n3", name="0", type="Value", description="0")

    if "label" in Node.model_fields:
        node = Node(id="n1", name="Qubit", type="Concept", description="q", label="Concept")
    edge = Edge(source_node_id="n1", target_node_id="n1", relationship_name="related_to")
    edge1 = Edge(source_node_id="n1", target_node_id="n2", relationship_name="takes_values")
    edge2 = Edge(source_node_id="n1", target_node_id="n3", relationship_name="takes_values")

    if "summary" in KnowledgeGraph.model_fields:
        return KnowledgeGraph(summary="", description="", nodes=[node], edges=[edge])
    return KnowledgeGraph(nodes=[node, node1, node2], edges=[edge, edge1, edge2])


def _make_ontology_resolver():
    ontology_xml = """<?xml version="1.0"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/Concept"/>
  <owl:Class rdf:about="http://example.org/Value">
    <rdfs:subClassOf rdf:resource="http://example.org/Concept"/>
  </owl:Class>
</rdf:RDF>
"""
    return RDFLibOntologyResolver(StringIO(ontology_xml))


def _normalize_document_contains(chunk):
    normalized = set()
    for edge, entity in chunk.contains or []:
        if not isinstance(entity, Entity) or isinstance(entity, EntityType):
            continue
        type_name = entity.is_a.name if entity.is_a else None
        normalized.add((edge.relationship_type, entity.name, type_name))
    return normalized


def _normalize_edges_default(edges, id_to_name):
    normalized = set()
    for source_id, target_id, rel, _props in edges:
        source_name = id_to_name.get(source_id)
        target_name = id_to_name.get(target_id)
        if source_name and target_name:
            normalized.add((source_name, rel, target_name))
    return normalized


def _normalize_edges_poc(chunk):
    normalized = set()
    for _edge, entity in chunk.contains or []:
        for rel_edge, target in getattr(entity, "relations", []):
            normalized.add((entity.name, rel_edge.relationship_type, target.name))
    return normalized


def test_expand_vs_poc_expand_populates_same_entities():
    chunk_for_expand = _make_chunk()
    chunk_for_poc = _make_chunk()
    chunk_graphs = [_kg()]

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        [chunk_for_expand],
        chunk_graphs,
    )

    poc_expand_with_nodes_and_edges(
        [chunk_for_poc],
        chunk_graphs,
    )

    assert _normalize_document_contains(chunk_for_poc) == _normalize_document_contains(
        chunk_for_expand
    )

    id_to_name = {
        entity.id: entity.name
        for _edge, entity in (chunk_for_expand.contains or [])
        if isinstance(entity, Entity) or isinstance(entity, EntityType)
    }

    assert _normalize_edges_default(graph_edges, id_to_name) == _normalize_edges_poc(chunk_for_poc)


def test_expand_vs_poc_expand_populates_same_entities_with_ontology():
    chunk_for_expand = _make_chunk()
    chunk_for_poc = _make_chunk()
    chunk_graphs = [_kg()]
    resolver = _make_ontology_resolver()

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        [chunk_for_expand],
        chunk_graphs,
        resolver,
        {},
    )

    poc_expand_with_nodes_and_edges(
        [chunk_for_poc],
        chunk_graphs,
        resolver,
        {},
    )

    assert _normalize_document_contains(chunk_for_poc) == _normalize_document_contains(
        chunk_for_expand
    )

    id_to_name = {
        entity.id: entity.name
        for _edge, entity in (chunk_for_expand.contains or [])
        if isinstance(entity, Entity) or isinstance(entity, EntityType)
    }

    assert _normalize_edges_default(graph_edges, id_to_name) == _normalize_edges_poc(chunk_for_poc)
