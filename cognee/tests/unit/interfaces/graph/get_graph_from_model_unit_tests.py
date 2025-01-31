import pytest
from typing import List, Optional
from cognee.infrastructure.engine import DataPoint

from cognee.modules.graph.utils import get_graph_from_model


class Document(DataPoint):
    path: str
    _metadata = {"index_fields": [], "type": "Document"}


class DocumentChunk(DataPoint):
    part_of: Document
    text: str
    contains: List["Entity"] = None
    _metadata = {"index_fields": ["text"], "type": "DocumentChunk"}


class EntityType(DataPoint):
    name: str
    _metadata = {"index_fields": ["name"], "type": "EntityType"}


class Entity(DataPoint):
    name: str
    is_type: EntityType
    _metadata = {"index_fields": ["name"], "type": "Entity"}


DocumentChunk.model_rebuild()


@pytest.mark.asyncio
async def test_get_graph_from_model_simple_structure():
    """
    Tests simple pydantic structure for get_graph_from_model
    """

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
