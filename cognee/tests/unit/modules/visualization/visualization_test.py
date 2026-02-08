import pytest
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)


@pytest.mark.asyncio
async def test_create_cognee_style_network_with_logo():
    nodes_data = [
        (
            1,
            {
                "type": "Entity",
                "name": "Node1",
                "updated_at": 123,
                "created_at": 123,
                "source_task": "extract_graph_from_data",
                "source_pipeline": "cognify_pipeline",
                "source_note_set": "research_notes",
            },
        ),
        (
            2,
            {
                "type": "DocumentChunk",
                "name": "Node2",
                "updated_at": 123,
                "created_at": 123,
                "source_task": "extract_chunks_from_documents",
                "source_pipeline": "cognify_pipeline",
                "source_note_set": "meeting_notes",
            },
        ),
    ]
    edges_data = [
        (1, 2, "related_to", {}),
    ]
    graph_data = (nodes_data, edges_data)

    html_output = await cognee_network_visualization(graph_data)

    assert isinstance(html_output, str)

    assert "<html" in html_output
    assert "d3.v7.min.js" in html_output
    assert "var nodes =" in html_output
    assert "var links =" in html_output

    # Phase 1: Zoom controls
    assert 'id="btn-fit"' in html_output
    assert 'id="btn-zoom-in"' in html_output
    assert 'id="btn-zoom-out"' in html_output
    assert "smoothZoomTo" in html_output
    assert "zoomToFit" in html_output
    assert "Math.exp(" in html_output

    # Phase 2: Density visualization
    assert 'data-layer="heatmap"' in html_output
    assert 'data-layer="typeclouds"' in html_output
    assert "heatmapGrid" in html_output
    assert "typeGrids" in html_output

    # Phase 3: Performance features
    assert "quadtree" in html_output
    assert "getViewportWorld" in html_output
    assert "isInViewport" in html_output
    assert "useDots" in html_output

    # Phase 3g: Loading overlay
    assert 'id="loading-overlay"' in html_output
    assert 'id="loading-bar"' in html_output

    # Phase 4: Minimap and FPS
    assert 'id="minimap-canvas"' in html_output
    assert 'id="fps-counter"' in html_output

    # Phase 5: Provenance tracking
    assert 'data-colorby="type"' in html_output
    assert 'data-colorby="task"' in html_output
    assert 'data-colorby="pipeline"' in html_output
    assert "colorByMode" in html_output
    assert "recolorNodes" in html_output
    assert "taskColors" in html_output
    assert "pipelineColors" in html_output
    assert "notesetColors" in html_output
    assert "Source Task" in html_output
    assert "Source Pipeline" in html_output
    assert "Source Note Set" in html_output
    assert 'data-colorby="noteset"' in html_output
    assert "updateLegend" in html_output
