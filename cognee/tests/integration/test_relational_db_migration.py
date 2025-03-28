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

    #1. Search the graph
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Tell me about the artist AC/DC"
    )
    print("Search results:", search_results)

    #2. Assert that the search results contain "AC/DC"
    assert any("AC/DC" in r for r in search_results), "AC/DC not found in search results!"

    #3. Directly verify the 'ReportsTo' hierarchy
    db_provider = os.getenv("GRAPH_DATABASE_PROVIDER", "networkx").lower()

    distinct_node_names = set()
    found_edges = set()

    if db_provider == "neo4j":
        query_str = """
        MATCH (n)-[r:ReportsTo]->(m)
        RETURN n, r, m
        """
        rows = await graph_engine.query(query_str)
        for row in rows:
            n_data = row["n"]
            r_data = row["r"]  
            m_data = row["m"]

            source_name = n_data["name"]
            target_name = m_data["name"]

            found_edges.add((source_name, target_name))
            distinct_node_names.add(source_name)
            distinct_node_names.add(target_name)

    elif db_provider == "kuzu":
        query_str = """
        MATCH (n:Node)-[r:EDGE]->(m:Node)
        WHERE r.relationship_name = 'ReportsTo'
        RETURN r, n, m
        """

        rows = await graph_engine.query(query_str)
        for row in rows:
            r_data = row[0]
            n_data = row[1]
            m_data = row[2]

            source_props = {}
            if "properties" in n_data and n_data["properties"]:
                source_props = json.loads(n_data["properties"])

            source_name = source_props.get("name", f"id:{n_data['id']}")

            target_props = {}
            if "properties" in m_data and m_data["properties"]:
                target_props = json.loads(m_data["properties"])

            target_name = target_props.get("name", f"id:{m_data['id']}")

            found_edges.add((source_name, target_name))
            distinct_node_names.add(source_name)
            distinct_node_names.add(target_name)

    elif db_provider == "networkx":
        nodes, edges = await graph_engine.get_graph_data()
        for (src, tgt, key, edge_data) in edges:
            if key == "ReportsTo":
                source_name = nodes_dict(nodes).get(src, {}).get("name", None)
                target_name = nodes_dict(nodes).get(tgt, {}).get("name", None)
                if source_name and target_name:
                    found_edges.add((source_name, target_name))
                    distinct_node_names.add(source_name)
                    distinct_node_names.add(target_name)

    assert len(distinct_node_names) == 8, f"Expected 8 distinct node references, found {len(distinct_node_names)}"

    assert len(found_edges) == 7, f"Expected 7 'ReportsTo' edges, got {len(found_edges)}"

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
        assert e in found_edges, f"Edge {e} not found in the actual 'ReportsTo' edges!"

    print(f"All checks passed for {db_provider} with single-edge-query approach.")


def nodes_dict(nodes):
    """
    Helper for the NetworkX branch:
    Takes a list of (node_id, data) and returns {node_id: data}.
    """
    return {n_id: data for (n_id, data) in nodes}
