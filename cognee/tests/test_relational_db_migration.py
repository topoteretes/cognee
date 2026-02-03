import pathlib
import os
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
from cognee.modules.users.methods import get_default_user
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

TEST_DATASET_NAME = "migration_test_dataset"


def nodes_dict(nodes):
    return {n_id: data for (n_id, data) in nodes}


def normalize_node_name(node_name: str) -> str:
    if node_name and ":" in node_name:
        prefix, suffix = node_name.split(":", 1)
        prefix = prefix.capitalize()
        return f"{prefix}:{suffix}"
    return node_name


async def setup_test_db():
    # Disable backend access control to migrate relational data
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    user = await get_default_user()
    await create_authorized_dataset(TEST_DATASET_NAME, user)

    migration_engine = get_migration_relational_engine()
    return migration_engine


async def relational_db_migration():
    migration_engine = await setup_test_db()
    schema = await migration_engine.extract_schema()

    graph_engine = await get_graph_engine()
    await migrate_relational_database(graph_engine, schema=schema)

    # 1. Search the graph
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Tell me about the artist AC/DC",
        datasets=[TEST_DATASET_NAME],
    )
    print("Search results:", search_results)

    # 2. Assert that the search results contain "AC/DC"
    assert any("AC/DC" in r for r in search_results), "AC/DC not found in search results!"

    migration_db_provider = migration_engine.engine.dialect.name
    if migration_db_provider == "postgresql":
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

            source_name = normalize_node_name(n_data.get("name", ""))
            target_name = normalize_node_name(m_data.get("name", ""))

            if source_name and target_name:
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
        raise ValueError(f"Unsupported graph database provider: {graph_db_provider}")

    assert len(distinct_node_names) == 12, (
        f"Expected 12 distinct node references, found {len(distinct_node_names)}"
    )
    assert len(found_edges) == 15, f"Expected 15 {relationship_label} edges, got {len(found_edges)}"

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
    if migration_db_provider == "sqlite":
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
            node_count = rows_n[0][0]

            query_edges = "MATCH (n:Node)-[r:EDGE]->(m:Node) RETURN count(r) as c"
            rows_e = await graph_engine.query(query_edges)
            edge_count = rows_e[0][0]

        elif graph_db_provider == "networkx":
            nodes, edges = await graph_engine.get_graph_data()
            node_count = len(nodes)
            edge_count = len(edges)

        # NOTE: Because of the different size of the postgres and sqlite databases,
        #       different number of nodes and edges are expected
        assert node_count == 543, f"Expected 543 nodes, got {node_count}"
        assert edge_count == 1317, f"Expected 1317 edges, got {edge_count}"

    elif migration_db_provider == "postgresql":
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
            node_count = rows_n[0][0]

            query_edges = "MATCH (n:Node)-[r:EDGE]->(m:Node) RETURN count(r) as c"
            rows_e = await graph_engine.query(query_edges)
            edge_count = rows_e[0][0]

        elif graph_db_provider == "networkx":
            nodes, edges = await graph_engine.get_graph_data()
            node_count = len(nodes)
            edge_count = len(edges)

        # NOTE: Because of the different size of the postgres and sqlite databases,
        #       different number of nodes and edges are expected
        assert node_count == 522, f"Expected 522 nodes, got {node_count}"
        assert edge_count == 961, f"Expected 961 edges, got {edge_count}"

    print(f"Node & edge count validated: node_count={node_count}, edge_count={edge_count}.")

    print(f"All checks passed for {graph_db_provider} provider with '{relationship_label}' edges!")


async def test_schema_only_migration():
    # 1. Setup test DB and extract schema
    migration_engine = await setup_test_db()
    schema = await migration_engine.extract_schema()

    # 2. Setup graph engine
    graph_engine = await get_graph_engine()

    # 4. Migrate schema only
    await migrate_relational_database(graph_engine, schema=schema, schema_only=True)

    # 5. Verify number of tables through search
    search_results = await cognee.search(
        query_text="How many tables are there in this database",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        top_k=30,
        datasets=[TEST_DATASET_NAME],
    )
    assert any("11" in r for r in search_results), (
        "Number of tables in the database reported in search_results is either None or not equal to 11"
    )

    graph_db_provider = os.getenv("GRAPH_DATABASE_PROVIDER", "networkx").lower()

    edge_counts = {
        "is_part_of": 0,
        "has_relationship": 0,
        "foreign_key": 0,
    }

    if graph_db_provider == "neo4j":
        for rel_type in edge_counts.keys():
            query_str = f"""
            MATCH ()-[r:{rel_type}]->()
            RETURN count(r) as c
            """
            rows = await graph_engine.query(query_str)
            edge_counts[rel_type] = rows[0]["c"]

    elif graph_db_provider == "kuzu":
        for rel_type in edge_counts.keys():
            query_str = f"""
            MATCH ()-[r:EDGE]->()
            WHERE r.relationship_name = '{rel_type}'
            RETURN count(r) as c
            """
            rows = await graph_engine.query(query_str)
            edge_counts[rel_type] = rows[0][0]

    elif graph_db_provider == "networkx":
        nodes, edges = await graph_engine.get_graph_data()
        for _, _, key, _ in edges:
            if key in edge_counts:
                edge_counts[key] += 1

    else:
        raise ValueError(f"Unsupported graph database provider: {graph_db_provider}")

    # 7. Assert counts match expected values
    expected_counts = {
        "is_part_of": 11,
        "has_relationship": 22,
        "foreign_key": 11,
    }

    for rel_type, expected in expected_counts.items():
        actual = edge_counts[rel_type]
        assert actual == expected, (
            f"Expected {expected} edges for relationship '{rel_type}', but found {actual}"
        )

    print("Schema-only migration edge counts validated successfully!")
    print(f"Edge counts: {edge_counts}")


async def test_search_result_quality():
    from cognee.infrastructure.databases.relational import (
        get_migration_relational_engine,
    )

    user = await get_default_user()
    await create_authorized_dataset(TEST_DATASET_NAME, user)

    # Get relational database with original data
    migration_engine = get_migration_relational_engine()
    from sqlalchemy import text

    async with migration_engine.engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT
                    c.CustomerId,
                    c.FirstName,
                    c.LastName,
                    GROUP_CONCAT(i.InvoiceId, ',') AS invoice_ids
                FROM Customer AS c
                LEFT JOIN Invoice AS i ON c.CustomerId = i.CustomerId
                GROUP BY c.CustomerId, c.FirstName, c.LastName
            """)
        )

        for row in result:
            # Get expected invoice IDs from relational DB for each Customer
            customer_id = row.CustomerId
            invoice_ids = row.invoice_ids.split(",") if row.invoice_ids else []
            print(f"Relational DB Customer {customer_id}: {invoice_ids}")

            # Use Cognee search to get invoice IDs for the same Customer but by providing Customer name
            search_results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=f"List me all the invoices of Customer:{row.FirstName} {row.LastName}.",
                top_k=50,
                system_prompt="Just return me the invoiceID as a number without any text. This is an example output: ['1', '2', '3']. Where 1, 2, 3 are invoiceIDs of an invoice",
                datasets=[TEST_DATASET_NAME],
            )
            print(f"Cognee search result: {search_results}")

            import ast

            lst = ast.literal_eval(search_results[0])  # converts string -> Python list
            # Transfrom both lists to int for comparison, sorting and type consistency
            lst = sorted([int(x) for x in lst])
            invoice_ids = sorted([int(x) for x in invoice_ids])
            assert lst == invoice_ids, (
                f"Search results {lst} do not match expected invoice IDs {invoice_ids} for Customer:{customer_id}"
            )


async def test_migration_sqlite():
    database_to_migrate_path = os.path.join(pathlib.Path(__file__).parent, "test_data/")

    cognee.config.set_migration_db_config(
        {
            "migration_db_path": database_to_migrate_path,
            "migration_db_name": "migration_database.sqlite",
            "migration_db_provider": "sqlite",
        }
    )

    await relational_db_migration()
    await test_search_result_quality()
    await test_schema_only_migration()


async def test_migration_postgres():
    # To run test manually you first need to run the Chinook_PostgreSql.sql script in the test_data directory
    cognee.config.set_migration_db_config(
        {
            "migration_db_name": "test_migration_db",
            "migration_db_host": "127.0.0.1",
            "migration_db_port": "5432",
            "migration_db_username": "cognee",
            "migration_db_password": "cognee",
            "migration_db_provider": "postgres",
        }
    )
    await relational_db_migration()
    await test_schema_only_migration()


async def main():
    print("Starting SQLite database migration test...")
    await test_migration_sqlite()
    print("Starting PostgreSQL database migration test...")
    await test_migration_postgres()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
