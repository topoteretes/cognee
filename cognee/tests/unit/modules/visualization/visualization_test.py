import pytest
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)


@pytest.mark.asyncio
async def test_create_cognee_style_network_with_logo():
    nodes_data = [
        (1, {"type": "Entity", "name": "Node1", "updated_at": 123, "created_at": 123}),
        (
            2,
            {
                "type": "DocumentChunk",
                "name": "Node2",
                "updated_at": 123,
                "created_at": 123,
            },
        ),
    ]
    edges_data = [
        (1, 2, "related_to", {}),
    ]
    graph_data = (nodes_data, edges_data)

    html_output = await cognee_network_visualization(graph_data)

    assert isinstance(html_output, str)

    assert "<html>" in html_output
    assert '<script src="https://d3js.org/d3.v5.min.js"></script>' in html_output
    assert "var nodes =" in html_output
    assert "var links =" in html_output
