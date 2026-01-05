# cognee/tests/test_usage_frequency.py
"""
Test suite for usage frequency tracking functionality.

Tests cover:
- Frequency extraction from CogneeUserInteraction nodes
- Time window filtering
- Frequency weight application to graph
- Edge cases and error handling
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from cognee.tasks.memify.extract_usage_frequency import (
    extract_usage_frequency,
    add_frequency_weights,
    create_usage_frequency_pipeline,
    run_usage_frequency_update,
)
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge


def create_mock_node(node_id: str, attributes: Dict[str, Any]) -> Node:
    """Helper to create mock Node objects."""
    node = Node(node_id, attributes)
    return node


def create_mock_edge(node1: Node, node2: Node, relationship_type: str, attributes: Dict[str, Any] = None) -> Edge:
    """Helper to create mock Edge objects."""
    edge_attrs = attributes or {}
    edge_attrs['relationship_type'] = relationship_type
    edge = Edge(node1, node2, attributes=edge_attrs, directed=True)
    return edge


def create_interaction_graph(
    interaction_count: int = 3,
    target_nodes: list = None,
    time_offset_hours: int = 0
) -> CogneeGraph:
    """
    Create a mock CogneeGraph with interaction nodes.
    
    :param interaction_count: Number of interactions to create
    :param target_nodes: List of target node IDs to reference
    :param time_offset_hours: Hours to offset timestamp (negative = past)
    :return: CogneeGraph with mocked interaction data
    """
    graph = CogneeGraph(directed=True)
    
    if target_nodes is None:
        target_nodes = ['node1', 'node2', 'node3']
    
    # Create some target graph element nodes
    element_nodes = {}
    for i, node_id in enumerate(target_nodes):
        element_node = create_mock_node(
            node_id,
            {
                'type': 'DocumentChunk',
                'text': f'This is content for {node_id}',
                'name': f'Element {i+1}'
            }
        )
        graph.add_node(element_node)
        element_nodes[node_id] = element_node
    
    # Create interaction nodes and edges
    timestamp = datetime.now() + timedelta(hours=time_offset_hours)
    
    for i in range(interaction_count):
        # Create interaction node
        interaction_id = f'interaction_{i}'
        target_id = target_nodes[i % len(target_nodes)]
        
        interaction_node = create_mock_node(
            interaction_id,
            {
                'type': 'CogneeUserInteraction',
                'timestamp': timestamp.isoformat(),
                'query_text': f'Sample query {i}',
                'target_node_id': target_id  # Also store in attributes for completeness
            }
        )
        graph.add_node(interaction_node)
        
        # Create edge from interaction to target element
        target_element = element_nodes[target_id]
        edge = create_mock_edge(
            interaction_node,
            target_element,
            'used_graph_element_to_answer',
            {'timestamp': timestamp.isoformat()}
        )
        graph.add_edge(edge)
    
    return graph


@pytest.mark.asyncio
async def test_extract_usage_frequency_basic():
    """Test basic frequency extraction with simple interaction data."""
    # Create mock graph with 3 interactions
    # node1 referenced twice, node2 referenced once
    mock_graph = create_interaction_graph(
        interaction_count=3,
        target_nodes=['node1', 'node1', 'node2']
    )
    
    # Extract frequencies
    result = await extract_usage_frequency(
        subgraphs=[mock_graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    # Assertions
    assert 'node_frequencies' in result
    assert 'edge_frequencies' in result
    assert result['node_frequencies']['node1'] == 2
    assert result['node_frequencies']['node2'] == 1
    assert result['total_interactions'] == 3
    assert result['interactions_in_window'] == 3


@pytest.mark.asyncio
async def test_extract_usage_frequency_time_window():
    """Test that time window filtering works correctly."""
    # Create two graphs: one recent, one old
    recent_graph = create_interaction_graph(
        interaction_count=2,
        target_nodes=['node1', 'node2'],
        time_offset_hours=-1  # 1 hour ago
    )
    
    old_graph = create_interaction_graph(
        interaction_count=2,
        target_nodes=['node3', 'node4'],
        time_offset_hours=-200  # 200 hours ago (> 7 days)
    )
    
    # Extract with 7-day window
    result = await extract_usage_frequency(
        subgraphs=[recent_graph, old_graph],
        time_window=timedelta(days=7),
        min_interaction_threshold=1
    )
    
    # Only recent interactions should be counted
    assert result['total_interactions'] == 4  # All interactions found
    assert result['interactions_in_window'] == 2  # Only recent ones counted
    assert 'node1' in result['node_frequencies']
    assert 'node2' in result['node_frequencies']
    assert 'node3' not in result['node_frequencies']  # Too old
    assert 'node4' not in result['node_frequencies']  # Too old


@pytest.mark.asyncio
async def test_extract_usage_frequency_threshold():
    """Test minimum interaction threshold filtering."""
    # Create graph where node1 has 3 interactions, node2 has 1
    mock_graph = create_interaction_graph(
        interaction_count=4,
        target_nodes=['node1', 'node1', 'node1', 'node2']
    )
    
    # Extract with threshold of 2
    result = await extract_usage_frequency(
        subgraphs=[mock_graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=2
    )
    
    # Only node1 should be in results (3 >= 2)
    assert 'node1' in result['node_frequencies']
    assert result['node_frequencies']['node1'] == 3
    assert 'node2' not in result['node_frequencies']  # Below threshold


@pytest.mark.asyncio
async def test_extract_usage_frequency_multiple_graphs():
    """Test extraction across multiple subgraphs."""
    graph1 = create_interaction_graph(
        interaction_count=2,
        target_nodes=['node1', 'node2']
    )
    
    graph2 = create_interaction_graph(
        interaction_count=2,
        target_nodes=['node1', 'node3']
    )
    
    result = await extract_usage_frequency(
        subgraphs=[graph1, graph2],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    # node1 should have frequency of 2 (once from each graph)
    assert result['node_frequencies']['node1'] == 2
    assert result['node_frequencies']['node2'] == 1
    assert result['node_frequencies']['node3'] == 1
    assert result['total_interactions'] == 4


@pytest.mark.asyncio
async def test_extract_usage_frequency_empty_graph():
    """Test handling of empty graphs."""
    empty_graph = CogneeGraph(directed=True)
    
    result = await extract_usage_frequency(
        subgraphs=[empty_graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    assert result['node_frequencies'] == {}
    assert result['edge_frequencies'] == {}
    assert result['total_interactions'] == 0
    assert result['interactions_in_window'] == 0


@pytest.mark.asyncio
async def test_extract_usage_frequency_invalid_timestamps():
    """Test handling of invalid timestamp formats."""
    graph = CogneeGraph(directed=True)
    
    # Create interaction with invalid timestamp
    bad_interaction = create_mock_node(
        'bad_interaction',
        {
            'type': 'CogneeUserInteraction',
            'timestamp': 'not-a-valid-timestamp',
            'target_node_id': 'node1'
        }
    )
    graph.add_node(bad_interaction)
    
    # Should not crash, just skip invalid interaction
    result = await extract_usage_frequency(
        subgraphs=[graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    assert result['total_interactions'] == 0  # Invalid interaction not counted


@pytest.mark.asyncio
async def test_extract_usage_frequency_element_type_tracking():
    """Test that element type frequencies are tracked."""
    graph = CogneeGraph(directed=True)
    
    # Create different types of target nodes
    chunk_node = create_mock_node('chunk1', {'type': 'DocumentChunk', 'text': 'content'})
    entity_node = create_mock_node('entity1', {'type': 'Entity', 'name': 'Alice'})
    
    graph.add_node(chunk_node)
    graph.add_node(entity_node)
    
    # Create interactions pointing to each
    timestamp = datetime.now().isoformat()
    
    for i, target in enumerate([chunk_node, chunk_node, entity_node]):
        interaction = create_mock_node(
            f'interaction_{i}',
            {'type': 'CogneeUserInteraction', 'timestamp': timestamp}
        )
        graph.add_node(interaction)
        
        edge = create_mock_edge(interaction, target, 'used_graph_element_to_answer')
        graph.add_edge(edge)
    
    result = await extract_usage_frequency(
        subgraphs=[graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    # Check element type frequencies
    assert 'element_type_frequencies' in result
    assert result['element_type_frequencies']['DocumentChunk'] == 2
    assert result['element_type_frequencies']['Entity'] == 1


@pytest.mark.asyncio
async def test_add_frequency_weights():
    """Test adding frequency weights to graph via adapter."""
    # Mock graph adapter
    mock_adapter = AsyncMock()
    mock_adapter.get_node_by_id = AsyncMock(return_value={
        'id': 'node1',
        'properties': {'type': 'DocumentChunk', 'text': 'content'}
    })
    mock_adapter.update_node_properties = AsyncMock()
    
    # Mock usage frequencies
    usage_frequencies = {
        'node_frequencies': {'node1': 5, 'node2': 3},
        'edge_frequencies': {},
        'last_processed_timestamp': datetime.now().isoformat()
    }
    
    # Add weights
    await add_frequency_weights(mock_adapter, usage_frequencies)
    
    # Verify adapter methods were called
    assert mock_adapter.get_node_by_id.call_count == 2
    assert mock_adapter.update_node_properties.call_count == 2
    
    # Verify the properties passed to update include frequency_weight
    calls = mock_adapter.update_node_properties.call_args_list
    properties_updated = calls[0][0][1]  # Second argument of first call
    assert 'frequency_weight' in properties_updated
    assert properties_updated['frequency_weight'] == 5


@pytest.mark.asyncio
async def test_add_frequency_weights_node_not_found():
    """Test handling when node is not found in graph."""
    mock_adapter = AsyncMock()
    mock_adapter.get_node_by_id = AsyncMock(return_value=None)  # Node not found
    mock_adapter.update_node_properties = AsyncMock()
    
    usage_frequencies = {
        'node_frequencies': {'nonexistent_node': 5},
        'edge_frequencies': {},
        'last_processed_timestamp': datetime.now().isoformat()
    }
    
    # Should not crash
    await add_frequency_weights(mock_adapter, usage_frequencies)
    
    # Update should not be called since node wasn't found
    assert mock_adapter.update_node_properties.call_count == 0


@pytest.mark.asyncio
async def test_add_frequency_weights_with_metadata_support():
    """Test that metadata is stored when adapter supports it."""
    mock_adapter = AsyncMock()
    mock_adapter.get_node_by_id = AsyncMock(return_value={'properties': {}})
    mock_adapter.update_node_properties = AsyncMock()
    mock_adapter.set_metadata = AsyncMock()  # Adapter supports metadata
    
    usage_frequencies = {
        'node_frequencies': {'node1': 5},
        'edge_frequencies': {},
        'element_type_frequencies': {'DocumentChunk': 5},
        'total_interactions': 10,
        'interactions_in_window': 8,
        'last_processed_timestamp': datetime.now().isoformat()
    }
    
    await add_frequency_weights(mock_adapter, usage_frequencies)
    
    # Verify metadata was stored
    mock_adapter.set_metadata.assert_called_once()
    metadata_key, metadata_value = mock_adapter.set_metadata.call_args[0]
    assert metadata_key == 'usage_frequency_stats'
    assert 'total_interactions' in metadata_value
    assert metadata_value['total_interactions'] == 10


@pytest.mark.asyncio
async def test_create_usage_frequency_pipeline():
    """Test pipeline creation returns correct task structure."""
    mock_adapter = AsyncMock()
    
    extraction_tasks, enrichment_tasks = await create_usage_frequency_pipeline(
        graph_adapter=mock_adapter,
        time_window=timedelta(days=7),
        min_interaction_threshold=2,
        batch_size=50
    )
    
    # Verify task structure
    assert len(extraction_tasks) == 1
    assert len(enrichment_tasks) == 1
    
    # Verify extraction task
    extraction_task = extraction_tasks[0]
    assert hasattr(extraction_task, 'function')
    
    # Verify enrichment task
    enrichment_task = enrichment_tasks[0]
    assert hasattr(enrichment_task, 'function')


@pytest.mark.asyncio
async def test_run_usage_frequency_update_integration():
    """Test the full end-to-end update process."""
    # Create mock graph with interactions
    mock_graph = create_interaction_graph(
        interaction_count=5,
        target_nodes=['node1', 'node1', 'node2', 'node3', 'node1']
    )
    
    # Mock adapter
    mock_adapter = AsyncMock()
    mock_adapter.get_node_by_id = AsyncMock(return_value={'properties': {}})
    mock_adapter.update_node_properties = AsyncMock()
    
    # Run the full update
    stats = await run_usage_frequency_update(
        graph_adapter=mock_adapter,
        subgraphs=[mock_graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    # Verify stats
    assert stats['total_interactions'] == 5
    assert stats['node_frequencies']['node1'] == 3
    assert stats['node_frequencies']['node2'] == 1
    assert stats['node_frequencies']['node3'] == 1
    
    # Verify adapter was called to update nodes
    assert mock_adapter.update_node_properties.call_count == 3  # 3 unique nodes


@pytest.mark.asyncio
async def test_extract_usage_frequency_no_used_graph_element_edges():
    """Test handling when there are interactions but no proper edges."""
    graph = CogneeGraph(directed=True)
    
    # Create interaction node
    interaction = create_mock_node(
        'interaction1',
        {
            'type': 'CogneeUserInteraction',
            'timestamp': datetime.now().isoformat(),
            'target_node_id': 'node1'
        }
    )
    graph.add_node(interaction)
    
    # Don't add any edges - interaction is orphaned
    
    result = await extract_usage_frequency(
        subgraphs=[graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    # Should find the interaction but no frequencies (no edges)
    assert result['total_interactions'] == 1
    assert result['node_frequencies'] == {}


@pytest.mark.asyncio
async def test_extract_usage_frequency_alternative_timestamp_field():
    """Test that 'created_at' field works as fallback for timestamp."""
    graph = CogneeGraph(directed=True)
    
    target = create_mock_node('target1', {'type': 'DocumentChunk'})
    graph.add_node(target)
    
    # Use 'created_at' instead of 'timestamp'
    interaction = create_mock_node(
        'interaction1',
        {
            'type': 'CogneeUserInteraction',
            'created_at': datetime.now().isoformat()  # Alternative field
        }
    )
    graph.add_node(interaction)
    
    edge = create_mock_edge(interaction, target, 'used_graph_element_to_answer')
    graph.add_edge(edge)
    
    result = await extract_usage_frequency(
        subgraphs=[graph],
        time_window=timedelta(days=1),
        min_interaction_threshold=1
    )
    
    # Should still work with created_at
    assert result['total_interactions'] == 1
    assert 'target1' in result['node_frequencies']


def test_imports():
    """Test that all required modules can be imported."""
    from cognee.tasks.memify.extract_usage_frequency import (
        extract_usage_frequency,
        add_frequency_weights,
        create_usage_frequency_pipeline,
        run_usage_frequency_update,
    )
    
    assert extract_usage_frequency is not None
    assert add_frequency_weights is not None
    assert create_usage_frequency_pipeline is not None
    assert run_usage_frequency_update is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])