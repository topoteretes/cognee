import json
import pytest
import pytest_asyncio
import os
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from cognee.tasks.ingestion import migrate_relational_database
from cognee.modules.search.types import SearchType
import cognee


def nodes_dict(nodes):
    return {n_id: data for (n_id, data) in nodes}


def normalize_node_name(node_name: str) -> str:
    if node_name and ":" in node_name:
        prefix, suffix = node_name.split(":", 1)
        prefix = prefix.capitalize()
        return f"{prefix}:{suffix}"
    return node_name


@pytest_asyncio.fixture()
async def setup_test_db():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    relational_engine = get_migration_relational_engine()
    return relational_engine


@pytest.mark.asyncio
async def test_relational_db_migration(setup_test_db):
    relational_engine = setup_test_db
    schema = await relational_engine.extract_schema()

    graph_engine = await get_graph_engine()
    await migrate_relational_database(graph_engine, schema=schema)

    # 1. Search the graph
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="Tell me about the artist AC/DC"
    )
    print("Search results:", search_results)

    # 2. Assert that the search results contain "AC/DC"
    assert any("AC/DC" in r for r in search_results), "AC/DC not found in search results!"

    relational_db_provider = os.getenv("MIGRATION_DB_PROVIDER", "sqlite").lower()
    if relational_db_provider == "postgres":
        relationship_label = "reports_to"
    else:
        relationship_label = "ReportsTo"

    # 3. Directly verify the 'reports to' hierarchy
    graph_db_provider = os.getenv("GRAPH_DATABASE_PROVIDER", "networkx").lower()

    distinct_node_names = set()
    found_edges = set()

    if graph_db_provider == "neo4j":
        query_str = f"""
        MATCH (n)-[r:{relationship_label}]->(m)
        RETURN n, r, m
        """
        rows = await graph_engine.query(query_str)
        for row in rows:
            n_data = row["n"]
            m_data = row["m"]

            source_name = normalize_node_name(n_data.get("name", ""))
            target_name = normalize_node_name(m_data.get("name", ""))

            found_edges.add((source_name, target_name))
            distinct_node_names.update([source_name, target_name])

    elif graph_db_provider == "kuzu":
        query_str = f"""
        MATCH (n:Node)-[r:EDGE]->(m:Node)
        WHERE r.relationship_name = '{relationship_label}'
        RETURN r, n, m
        """
        rows = await graph_engine.query(query_str)
        for row in rows:
            n_data = row[1]
            m_data = row[2]

            source_props = {}
            if "properties" in n_data and n_data["properties"]:
                source_props = json.loads(n_data["properties"])
            target_props = {}
            if "properties" in m_data and m_data["properties"]:
                target_props = json.loads(m_data["properties"])

            source_name = normalize_node_name(source_props.get("name", f"id:{n_data['id']}"))
            target_name = normalize_node_name(target_props.get("name", f"id:{m_data['id']}"))

            found_edges.add((source_name, target_name))
            distinct_node_names.update([source_name, target_name])

    elif graph_db_provider == "networkx":
        nodes, edges = await graph_engine.get_graph_data()
        node_map = nodes_dict(nodes)
        for src, tgt, key, edge_data in edges:
            if key == relationship_label:
                src_name = normalize_node_name(node_map[src].get("name"))
                tgt_name = normalize_node_name(node_map[tgt].get("name"))
                if src_name and tgt_name:
                    found_edges.add((src_name, tgt_name))
                    distinct_node_names.update([src_name, tgt_name])
    else:
        pytest.fail(f"Unsupported graph database provider: {graph_db_provider}")

    assert len(distinct_node_names) == 8, (
        f"Expected 8 distinct node references, found {len(distinct_node_names)}"
    )
    assert len(found_edges) == 7, f"Expected 7 {relationship_label} edges, got {len(found_edges)}"

    expected_edges = {
        ("Employee:5", "Employee:2"),
        ("Employee:2", "Employee:1"),
        ("Employee:4", "Employee:2"),
        ("Employee:6", "Employee:1"),
        ("Employee:8", "Employee:6"),
        ("Employee:7", "Employee:6"),
        ("Employee:3", "Employee:2"),
    }
    for e in expected_edges:
        assert e in found_edges, f"Edge {e} not found in the actual '{relationship_label}' edges!"

    # 4. Verify the total number of nodes and edges in the graph
    if relational_db_provider == "sqlite":
        if graph_db_provider == "neo4j":
            query_str = """
            MATCH (n)
            WITH count(n) AS node_count
            MATCH ()-[r]->()
            RETURN node_count, count(r) AS edge_count
            """
            rows = await graph_engine.query(query_str)
            node_count = rows[0]["node_count"]
            edge_count = rows[0]["edge_count"]

        elif graph_db_provider == "kuzu":
            query_nodes = "MATCH (n:Node) RETURN count(n) as c"
            rows_n = await graph_engine.query(query_nodes)
            node_count = rows_n[0]["c"]

            query_edges = "MATCH (n:Node)-[r:EDGE]->(m:Node) RETURN count(r) as c"
            rows_e = await graph_engine.query(query_edges)
            edge_count = rows_e[0]["c"]

        elif graph_db_provider == "networkx":
            nodes, edges = await graph_engine.get_graph_data()
            node_count = len(nodes)
            edge_count = len(edges)

        # NOTE: Because of the different size of the postgres and sqlite databases,
        #       different number of nodes and edges are expected
        assert node_count == 227, f"Expected 227 nodes, got {node_count}"
        assert edge_count == 580, f"Expected 580 edges, got {edge_count}"

    elif relational_db_provider == "postgres":
        if graph_db_provider == "neo4j":
            query_str = """
            MATCH (n)
            WITH count(n) AS node_count
            MATCH ()-[r]->()
            RETURN node_count, count(r) AS edge_count
            """
            rows = await graph_engine.query(query_str)
            node_count = rows[0]["node_count"]
            edge_count = rows[0]["edge_count"]

        elif graph_db_provider == "kuzu":
            query_nodes = "MATCH (n:Node) RETURN count(n) as c"
            rows_n = await graph_engine.query(query_nodes)
            node_count = rows_n[0]["c"]

            query_edges = "MATCH (n:Node)-[r:EDGE]->(m:Node) RETURN count(r) as c"
            rows_e = await graph_engine.query(query_edges)
            edge_count = rows_e[0]["c"]

        elif graph_db_provider == "networkx":
            nodes, edges = await graph_engine.get_graph_data()
            node_count = len(nodes)
            edge_count = len(edges)

        # NOTE: Because of the different size of the postgres and sqlite databases,
        #       different number of nodes and edges are expected
        assert node_count == 115, f"Expected 115 nodes, got {node_count}"
        assert edge_count == 356, f"Expected 356 edges, got {edge_count}"

    print(f"Node & edge count validated: node_count={node_count}, edge_count={edge_count}.")

    print(f"All checks passed for {graph_db_provider} provider with '{relationship_label}' edges!")
