"""Tests for SchemaTable and SchemaRelationship rendering in visualization (#5833)."""

import asyncio
import json
import re

import pytest

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from cognee.modules.visualization.preprocessor import preprocess


def test_preprocess_populates_rank_and_links_for_schema_tables():
    """Verify extract_schema_graph_data attaches rank to SchemaTable and preserves edges."""
    nodes_data = [
        (
            "t1",
            {"type": "SchemaTable", "name": "parts", "columns": [{"name": "id", "type": "uuid"}]},
        ),
        (
            "t2",
            {
                "type": "SchemaTable",
                "name": "pins",
                "columns": [{"name": "part_id", "type": "uuid"}],
            },
        ),
    ]
    edges_data = [("t2", "t1", "has_relationship", {"relationship_name": "references"})]

    result = preprocess((nodes_data, edges_data))
    schema_graph = result.schema_graph
    assert len(schema_graph["nodes"]) == 2
    for node in schema_graph["nodes"]:
        assert node["type"] == "SchemaTable"
        assert node["rank"] == 1
        assert len(node["fields"]) == 1

    assert len(schema_graph["links"]) == 1
    assert schema_graph["links"][0] == {"source": "t2", "target": "t1", "label": "references"}


def test_visualization_html_contains_schema_table_and_js_handlers(tmp_path):
    """Verify that generated HTML embeds SchemaTable nodes and schema_view.js handlers."""
    nodes_data = [
        (
            "t1",
            {"type": "SchemaTable", "name": "parts", "columns": [{"name": "id", "type": "uuid"}]},
        ),
        (
            "t2",
            {
                "type": "SchemaTable",
                "name": "pins",
                "columns": [{"name": "part_id", "type": "uuid"}],
            },
        ),
    ]
    edges_data = [("t2", "t1", "has_relationship", {"relationship_name": "references"})]

    out_file = str(tmp_path / "schema_graph.html")
    html = asyncio.run(cognee_network_visualization((nodes_data, edges_data), out_file))

    # Check embedded schemaGraphData
    match = re.search(r"const schemaGraphData = (\{.*?\});\n", html)
    assert match is not None, "schemaGraphData not embedded in html"
    data = json.loads(match.group(1))
    node_names = {n["name"] for n in data["nodes"]}
    assert "parts" in node_names
    assert "pins" in node_names
    assert len(data["links"]) == 1

    # Verify that schema_view.js contains the SchemaTable and link integration logic
    assert "n.type === 'SchemaTable'" in html
    assert "n.type === 'SchemaRelationship'" in html
    assert "schema_link_" in html
