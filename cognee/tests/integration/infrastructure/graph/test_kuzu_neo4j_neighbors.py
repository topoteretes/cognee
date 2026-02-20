import json
import os
from pathlib import Path

import pytest

import cognee

from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

DEMO_KG_PATH = os.path.join(os.path.dirname(__file__), "test_kg.json")


def _load_demo_kg() -> KnowledgeGraph:
    data = json.loads(Path(DEMO_KG_PATH).read_text(encoding="utf-8"))
    return KnowledgeGraph.model_validate(data)


def _extract_props(node: dict) -> dict:
    props = node.get("properties")
    if isinstance(props, str):
        try:
            return json.loads(props)
        except json.JSONDecodeError:
            return {}
    if isinstance(props, dict):
        return props
    return {}


def _normalize_neighbor(node: dict) -> dict:
    props = _extract_props(node)
    return {
        "id": str(node.get("id") or props.get("id") or ""),
        "name": node.get("name") or props.get("name") or "",
        "type": node.get("type") or props.get("type") or "",
        "description": node.get("description") or props.get("description") or "",
    }


@pytest.mark.asyncio
async def test_kuzu_neo4j_get_neighbors_match(tmp_path):
    pytest.importorskip("kuzu")
    pytest.importorskip("neo4j")

    neo4j_url = os.getenv("GRAPH_DATABASE_URL") or os.getenv("NEO4J_URL")
    if not neo4j_url:
        pytest.skip("Neo4j connection info not configured (GRAPH_DATABASE_URL/NEO4J_URL).")

    neo4j_user = os.getenv("GRAPH_DATABASE_USERNAME") or os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("GRAPH_DATABASE_PASSWORD") or os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("GRAPH_DATABASE_NAME") or os.getenv("NEO4J_DATABASE")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    kg = _load_demo_kg()
    if not kg.edges:
        pytest.skip("demo_kg.json has no edges to test neighbors.")

    center_node_id = kg.edges[0].source_node_id
    edge_rows = [
        (edge.source_node_id, edge.target_node_id, edge.relationship_name, {}) for edge in kg.edges
    ]

    kuzu_db_path = str(tmp_path / "kuzu_neighbors_graph")
    kuzu = KuzuAdapter(db_path=kuzu_db_path)
    neo4j = Neo4jAdapter(
        graph_database_url=neo4j_url,
        graph_database_username=neo4j_user,
        graph_database_password=neo4j_pass,
        graph_database_name=neo4j_db,
    )

    try:
        await kuzu.add_nodes(kg.nodes)
        await kuzu.add_edges(edge_rows)

        await neo4j.initialize()
        await neo4j.delete_graph()
        await neo4j.add_nodes(kg.nodes)
        await neo4j.add_edges(edge_rows)

        kuzu_neighbors = await kuzu.get_neighbors(center_node_id)
        neo4j_neighbors = await neo4j.get_neighbors(center_node_id)

        kuzu_norm = {
            (
                norm["id"],
                norm["name"],
                norm["type"],
                norm["description"],
            )
            for norm in (_normalize_neighbor(n) for n in kuzu_neighbors)
        }
        neo4j_norm = {
            (
                norm["id"],
                norm["name"],
                norm["type"],
                norm["description"],
            )
            for norm in (_normalize_neighbor(n) for n in neo4j_neighbors)
        }

        assert kuzu_norm == neo4j_norm
    finally:
        await neo4j.delete_graph()
        await kuzu.delete_graph()
