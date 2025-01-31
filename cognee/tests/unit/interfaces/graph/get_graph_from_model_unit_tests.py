import pytest
import asyncio
import random
from typing import List
from uuid import NAMESPACE_OID, uuid5
from uuid import uuid4

from IPython.utils.wildcard import is_type

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Entity import Entity, EntityType
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.modules.graph.utils import get_graph_from_model


@pytest.mark.asyncio
async def test_get_graph_from_model_basic_initialization():
    """Test the basic behavior of get_graph_from_model with a simple data point - without connection."""
    data_point = DataPoint(id=uuid4(), attributes={"name": "Node1"})
    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(
        data_point, added_nodes, added_edges, visited_properties
    )

    assert len(nodes) == 1
    assert len(edges) == 0
    assert str(data_point.id) in added_nodes


@pytest.mark.asyncio
async def test_get_graph_from_model_with_single_neighbor():
    """Test the behavior of get_graph_from_model when a data point has a single DataPoint property."""
    type_node = EntityType(
        id=uuid4(),
        name="Vehicle",
        description="This is a Vehicle node",
    )

    entity_node = Entity(
        id=uuid4(),
        name="Car",
        is_a=type_node,
        description="This is a car node",
    )
    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(
        entity_node, added_nodes, added_edges, visited_properties
    )

    assert len(nodes) == 2
    assert len(edges) == 1
    assert str(entity_node.id) in added_nodes
    assert str(type_node.id) in added_nodes
    assert (str(entity_node.id) + str(type_node.id) + "is_a") in added_edges


@pytest.mark.asyncio
async def test_get_graph_from_model_with_multiple_nested_connections():
    """Test the behavior of get_graph_from_model when a data point has multiple nested DataPoint property."""
    type_node = EntityType(
        id=uuid4(),
        name="Transportation tool",
        description="This is a Vehicle node",
    )

    entity_node_1 = Entity(
        id=uuid4(),
        name="Car",
        is_a=type_node,
        description="This is a car node",
    )

    entity_node_2 = Entity(
        id=uuid4(),
        name="Bus",
        is_a=type_node,
        description="This is a bus node",
    )

    document = Document(
        name="main_document", raw_data_location="home/", metadata_id=uuid4(), mime_type="test"
    )

    chunk = DocumentChunk(
        id=uuid4(),
        word_count=8,
        chunk_index=0,
        cut_type="test",
        text="The car and the bus are transportation tools",
        is_part_of=document,
        contains=[entity_node_1, entity_node_2],
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(chunk, added_nodes, added_edges, visited_properties)

    assert len(nodes) == 5
    assert len(edges) == 5

    assert str(entity_node_1.id) in added_nodes
    assert str(entity_node_2.id) in added_nodes
    assert str(type_node.id) in added_nodes
    assert str(document.id) in added_nodes
    assert str(chunk.id) in added_nodes

    assert (str(entity_node_1.id) + str(type_node.id) + "is_a") in added_edges
    assert (str(entity_node_2.id) + str(type_node.id) + "is_a") in added_edges
    assert (str(chunk.id) + str(document.id) + "is_part_of") in added_edges
    assert (str(chunk.id) + str(entity_node_1.id) + "contains") in added_edges
    assert (str(chunk.id) + str(entity_node_2.id) + "contains") in added_edges
