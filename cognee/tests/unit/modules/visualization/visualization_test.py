import pytest
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)


@pytest.mark.asyncio
async def test_create_cognee_style_network_with_logo(tmp_path):
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
                "source_node_set": "research_nodes",
                "source_user": "alice@example.com",
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
                "source_node_set": "meeting_nodes",
                "source_user": "bob@example.com",
            },
        ),
    ]
    edges_data = [
        (1, 2, "related_to", {}),
    ]
    graph_data = (nodes_data, edges_data)

    html_output = await cognee_network_visualization(
        graph_data, str(tmp_path / "graph_visualization.html")
    )

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

    # Ranked left-to-right layout
    assert 'data-layout="ranked"' in html_output
    assert 'data-layout="organic"' in html_output
    assert "computeRankedLayout" in html_output
    assert "computeOrganicLayout" in html_output
    assert "drawRankColumns" in html_output
    assert "drawOrganicClusters" in html_output
    assert '"type": "GraphNodeType"' in html_output
    assert '"type": "GraphRelationshipType"' in html_output

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
    assert "colorByMode" in html_output
    assert "recolorNodes" in html_output
    assert "taskColors" in html_output
    assert "pipelineColors" in html_output
    assert "nodesetColors" in html_output
    assert "userColors" in html_output
    # Inspector Provenance section labels (renamed from legacy
    # "Source Node Set"/"Source User" in the Phase 1e rewrite — the
    # provenance feature itself is asserted by data-colorby="nodeset"/"user"
    # and the inspectorSection presence below).
    assert '"Node Set"' in html_output
    assert '"User"' in html_output
    assert 'data-colorby="nodeset"' in html_output
    assert 'data-colorby="user"' in html_output
    assert "updateLegend" in html_output

    # Phase 1b: Label budget (key default + preprocessor-driven priority).
    # "Key" mode uses node.label_priority computed in preprocessor.py to
    # cap labels at documents / types / top-quartile-importance nodes.
    assert 'data-labelbudget="key"' in html_output
    assert 'data-labelbudget="all"' in html_output
    assert 'data-labelbudget="off"' in html_output
    assert "labelBudget" in html_output
    assert "label_priority" in html_output

    # Phase 1d: Pipeline (Story) layout — new default, bins nodes by
    # preprocessor.stage and lays them left-to-right in canonical order.
    assert 'data-layout="story"' in html_output
    assert "computePipelineLayout" in html_output
    assert "STORY_STAGE_ORDER" in html_output
    # Story is the new default initial layoutMode (replaces "ranked")
    assert 'var layoutMode="story"' in html_output

    # Phase 1e: Sectioned inspector with plain-English overview.
    # Reads preprocessor.stage / edge_class to build a "Source" chain and
    # a "Provenance" section that only shows when any field is set.
    assert "inspectorOverviewLine" in html_output
    assert "inspectorSection" in html_output
    assert "inspector-section-body" in html_output
    assert 'data-toggle="provenance"' in html_output or "provenance" in html_output

    # Phase 1 polish: keyboard nav, URL hash sync.
    # (Stage-tint palette was removed per user feedback — stages are
    # delimited by the right-edge gridline + the stage-label pill only.)
    assert "selectByKey" in html_output
    assert "readHashState" in html_output
    assert "writeHash" in html_output

    # Phase 1 polish: semantic Force mode tuning by edge_class.
    assert 'edge_class==="semantic"' in html_output

    # Phase 2: Schema rework — explicit-position ontology diagram is the
    # entire Schema tab (the bottom node-type-cards panel was removed
    # per user feedback; node-type detail lives on the boxes themselves).
    assert "renderSchemaDiagram" in html_output
    assert "sd-node-box" in html_output
    assert "sd-edge-path" in html_output
    assert "sd-edge-label" in html_output
    assert 'id="schema-selection"' in html_output  # focus / status bar
    assert "sd-mini-card" in html_output  # instance mini-cards
    assert "sd-spotlight" in html_output  # instance-connection spotlight

    # PR3: schema type inspector side panel + type→graph highlight bridge.
    # The DOM mount, the inspector JS that reads the PR2 contract fields,
    # and the highlight/tab-switch wiring must all be present (the inspector
    # view is no longer an empty stub).
    assert 'id="schema-side-panel"' in html_output
    assert "window._showSchemaInspector" in html_output
    assert "window._schemaTypeIndex" in html_output
    assert "window._highlightSchemaType" in html_output
    # PR2 contract field names read verbatim by the inspector.
    assert "sample_size" in html_output
    assert "relationships" in html_output
    assert "to_type" in html_output
    # Highlight action wiring + side-panel content markers.
    assert "Highlight in graph" in html_output
    assert 'data-action="highlight"' in html_output
    assert 'class="si-chip"' in html_output
    assert 'class="si-rel"' in html_output


@pytest.mark.asyncio
async def test_schema_tab_renders_schema_nodes_without_explicit_schema(tmp_path):
    nodes_data = [
        (
            "database",
            {
                "type": "DatabaseSchema",
                "name": "app",
                "database_type": "postgres",
                "tables": "{}",
                "sample_data": "{}",
                "description": "Application database",
            },
        ),
        (
            "users",
            {
                "type": "SchemaTable",
                "name": "users",
                "columns": '[{"name": "id", "type": "uuid"}, {"name": "email", "type": "text"}]',
                "primary_key": "id",
                "foreign_keys": "[]",
                "sample_rows": "[]",
                "row_count_estimate": 2,
                "description": "Users table",
            },
        ),
        (
            "posts",
            {
                "type": "SchemaTable",
                "name": "posts",
                "columns": '[{"name": "id", "type": "uuid"}, {"name": "user_id", "type": "uuid"}]',
                "primary_key": "id",
                "foreign_keys": "[]",
                "sample_rows": "[]",
                "row_count_estimate": 5,
                "description": "Posts table",
            },
        ),
        (
            "posts_user_fk",
            {
                "type": "SchemaRelationship",
                "name": "posts:user_id->users:id",
                "source_table": "posts",
                "target_table": "users",
                "relationship_type": "foreign_key",
                "source_column": "user_id",
                "target_column": "id",
                "description": "Posts user foreign key",
            },
        ),
    ]
    edges_data = [
        ("users", "database", "is_part_of", {"relationship_name": "is_part_of"}),
        ("posts", "database", "is_part_of", {"relationship_name": "is_part_of"}),
        ("posts", "posts_user_fk", "has_relationship", {"relationship_name": "foreign_key"}),
        (
            "posts_user_fk",
            "users",
            "has_relationship",
            {"relationship_name": "foreign_key"},
        ),
    ]

    html_output = await cognee_network_visualization(
        (nodes_data, edges_data), str(tmp_path / "schema_visualization.html")
    )

    assert "const schemaGraphData = " in html_output
    assert '"type": "SchemaTable"' in html_output
    assert '"name": "users"' in html_output
    assert '"label": "foreign_key"' in html_output
    # Phase 2 redesign — old `graphDataToSchemaGraph` model builder
    # replaced by `buildSchemaModel`; renderer still exposed as
    # `window._renderSchemaGraph` so the tab-switch handler keeps working.
    assert "buildSchemaModel" in html_output
    assert "window._renderSchemaGraph" in html_output
