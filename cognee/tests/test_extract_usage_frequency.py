# cognee/tests/test_usage_frequency.py
import pytest
import asyncio
from datetime import datetime, timedelta
from cognee.tasks.memify.extract_usage_frequency import extract_usage_frequency, add_frequency_weights

@pytest.mark.asyncio
async def test_extract_usage_frequency():
    # Mock CogneeGraph with user interactions
    mock_subgraphs = [{
        'nodes': [
            {
                'type': 'CogneeUserInteraction',
                'target_node_id': 'node1',
                'edge_type': 'viewed',
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'CogneeUserInteraction',
                'target_node_id': 'node1',
                'edge_type': 'viewed',
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'CogneeUserInteraction',
                'target_node_id': 'node2',
                'edge_type': 'referenced',
                'timestamp': datetime.now().isoformat()
            }
        ]
    }]

    # Test frequency extraction
    result = await extract_usage_frequency(
        mock_subgraphs, 
        time_window=timedelta(days=1), 
        min_interaction_threshold=1
    )

    assert 'node1' in result['node_frequencies']
    assert result['node_frequencies']['node1'] == 2
    assert result['edge_frequencies']['viewed'] == 2