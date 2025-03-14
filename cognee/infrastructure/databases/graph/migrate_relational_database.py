import logging
from decimal import Decimal
from sqlalchemy import text
from cognee.infrastructure.databases.relational.get_migration_relational_engine import (
    get_migration_relational_engine,
)

logger = logging.getLogger(__name__)


async def migrate_relational_database_cypher(graph_db, schema):
    """
    Migrates data from a relational database into a Neo4j graph.

    For each table in the schema:
      - Fetch all rows and create a node in Neo4j. The node id is built as "<table_name>:<primary_key_value>".
      - Create (or merge) a table node and link the row node to it via an "is_part_of" relationship.
    Then, for every foreign key defined in the schema:
      - Fetch the relationships from the relational DB and create a corresponding edge (using the foreign key column name as the relationship type)
        between the source and target row nodes.

    After the migration, returns the graph data (nodes and edges) as expected by the get_graph_data function.
    """
    engine = get_migration_relational_engine()

    async with engine.engine.begin() as cursor:
        # Migrate rows as nodes and connect them to their table node.
        for table_name, details in schema.items():
            # Fetch all rows for the current table.
            rows_result = await cursor.execute(text(f"SELECT * FROM {table_name};"))
            rows = rows_result.fetchall()

            for row in rows:
                # Build a dictionary of properties for this row.
                properties = {col["name"]: row[idx] for idx, col in enumerate(details["columns"])}

                # Convert Decimal values to float.
                for key in properties:
                    if isinstance(properties[key], Decimal):
                        properties[key] = float(properties[key])

                # Build the unique node id.
                if not details.get("primary_key"):
                    # Assume the first column in details['columns'] is the primary key.
                    node_id = f"{table_name}:{properties[details['columns'][0]['name']]}"
                else:
                    node_id = f"{table_name}:{properties[details['primary_key']]}"

                # Add extra properties.
                properties["label"] = table_name
                properties["type"] = "TableRow"

                async with graph_db.driver.session() as session:
                    # Create (or merge) the TableRow node.
                    await session.run(
                        """
                        MERGE (n:TableRow {id: $node_id})
                        SET n += $properties
                        """,
                        node_id=node_id,
                        properties=properties,
                    )
                    # Ensure the table node exists.
                    await session.run(
                        """
                        MERGE (t:Table {id: $table_name})
                        """,
                        table_name=table_name,
                    )
                    # Create the relationship from the row node to its table
                    # and set relationship properties for later extraction.
                    await session.run(
                        """
                        MATCH (n:TableRow {id: $node_id}), (t:Table {id: $table_name})
                        MERGE (n)-[r:is_part_of]->(t)
                        SET r.source_node_id = $node_id, r.target_node_id = $table_name
                        """,
                        node_id=node_id,
                        table_name=table_name,
                    )

        # Process foreign key relationships.
        for table_name, details in schema.items():
            for fk in details.get("foreign_keys", []):
                # Build aliases for the source and referenced tables.
                alias_1 = f"{table_name}_e1"
                alias_2 = f"{fk['ref_table']}_e2"

                # Determine the primary key column for the current table.
                if not details.get("primary_key"):
                    primary_key_col = details["columns"][0]["name"]
                else:
                    primary_key_col = details["primary_key"]

                # Build and execute the foreign key query.
                fk_query = text(
                    f"SELECT {alias_1}.{primary_key_col} AS source_id, "
                    f"{alias_2}.{fk['ref_column']} AS ref_value "
                    f"FROM {table_name} AS {alias_1} "
                    f"JOIN {fk['ref_table']} AS {alias_2} "
                    f"ON {alias_1}.{fk['column']} = {alias_2}.{fk['ref_column']};"
                )
                fk_result = await cursor.execute(fk_query)
                relations = fk_result.fetchall()

                for source_id, ref_value in relations:
                    # Construct node ids for the source and target rows.
                    source_node = f"{table_name}:{source_id}"
                    target_node = f"{fk['ref_table']}:{ref_value}"

                    async with graph_db.driver.session() as session:
                        # Create the relationship in Neo4j.
                        # Note: The relationship type is set to the foreign key column name.
                        await session.run(
                            f"""
                            MATCH (a:TableRow {{id: $source_node}}), (b:TableRow {{id: $target_node}})
                            MERGE (a)-[r:{fk["column"]}]->(b)
                            SET r.source_node_id = $source_node, r.target_node_id = $target_node
                            """,
                            source_node=source_node,
                            target_node=target_node,
                        )

    logger.info("Data populated into Neo4j successfully.")
    # Return the graph data (nodes, edges) using the provided get_graph_data function.
    return await graph_db.get_graph_data()
