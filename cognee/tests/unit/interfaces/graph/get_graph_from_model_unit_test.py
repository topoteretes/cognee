import pytest
from typing import List, Optional
from cognee.infrastructure.engine import DataPoint

from cognee.modules.graph.utils import get_graph_from_model


class Document(DataPoint):
    path: str
    metadata: dict = {"index_fields": []}


class DocumentChunk(DataPoint):
    part_of: Document
    text: str
    contains: List["Entity"] = None
    metadata: dict = {"index_fields": ["text"]}


class EntityType(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Entity(DataPoint):
    name: str
    is_type: EntityType
    metadata: dict = {"index_fields": ["name"]}


DocumentChunk.model_rebuild()


@pytest.mark.asyncio
async def test_get_graph_from_model_simple_structure():
    """Tests simple pydantic structure for get_graph_from_model"""

    entitytype = EntityType(
        name="TestType",
    )

    entity = Entity(name="TestEntity", is_type=entitytype)

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(entity, added_nodes, added_edges, visited_properties)

    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    assert len(edges) == 1, f"Expected 1 edges, got {len(edges)}"

    edge_key = str(entity.id) + str(entitytype.id) + "is_type"
    assert edge_key in added_edges, f"Edge {edge_key} not found"


@pytest.mark.asyncio
async def test_get_graph_from_model_with_document_and_chunk():
    """Tests multiple entities to document connection"""
    doc = Document(path="test/path")
    doc_chunk = DocumentChunk(part_of=doc, text="This is a chunk of text", contains=[])
    entity_type = EntityType(name="Person")
    entity = Entity(name="Alice", is_type=entity_type)
    entity2 = Entity(name="Alice2", is_type=entity_type)
    doc_chunk.contains.append(entity)
    doc_chunk.contains.append(entity2)

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(
        doc_chunk, added_nodes, added_edges, visited_properties
    )

    assert len(nodes) == 5, f"Expected 5 nodes, got {len(nodes)}"
    assert len(edges) == 5, f"Expected 5 edges, got {len(edges)}"


@pytest.mark.asyncio
async def test_get_graph_from_model_duplicate_references():
    """Tests duplicated objects in document list"""
    doc = Document(path="test/path")
    doc_chunk = DocumentChunk(part_of=doc, text="Chunk with duplicates", contains=[])

    entity_type = EntityType(name="Animal")
    shared_entity = Entity(name="Cat", is_type=entity_type)

    doc_chunk.contains.extend([shared_entity, shared_entity, shared_entity])

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(
        doc_chunk, added_nodes, added_edges, visited_properties
    )

    assert len(nodes) == 4, f"Expected 4 nodes, got {len(nodes)}"
    assert len(edges) == 3, f"Expected 3 edges, got {len(edges)}"


@pytest.mark.asyncio
async def test_get_graph_from_model_multi_level_nesting():
    """Tests multi level nested structure extraction"""
    doc = Document(path="multi-level/path")

    chunk1 = DocumentChunk(part_of=doc, text="Chunk 1 text", contains=[])
    chunk2 = DocumentChunk(part_of=doc, text="Chunk 2 text", contains=[])

    entity_type_vehicle = EntityType(name="Vehicle")
    entity_type_person = EntityType(name="Person")

    entity_car = Entity(name="Car", is_type=entity_type_vehicle)
    entity_bike = Entity(name="Bike", is_type=entity_type_vehicle)
    entity_alice = Entity(name="Alice", is_type=entity_type_person)

    chunk1.contains.extend([entity_car, entity_bike])
    chunk2.contains.append(entity_alice)

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(chunk1, added_nodes, added_edges, visited_properties)

    nodes2, edges2 = await get_graph_from_model(
        chunk2, added_nodes, added_edges, visited_properties
    )

    all_nodes = nodes + nodes2
    all_edges = edges + edges2

    assert len(all_nodes) == 8, f"Expected 8 nodes, got {len(all_nodes)}"
    assert len(all_edges) == 8, f"Expected 8 edges, got {len(all_edges)}"


@pytest.mark.asyncio
async def test_get_graph_from_model_no_contains():
    """Tests graph from model with empty contains element"""
    doc = Document(path="empty-contains/path")
    chunk = DocumentChunk(part_of=doc, text="A chunk with no entities", contains=[])

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(chunk, added_nodes, added_edges, visited_properties)

    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"
