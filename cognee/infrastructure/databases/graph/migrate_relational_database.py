import logging
from decimal import Decimal
from sqlalchemy import text
from cognee.infrastructure.databases.relational.get_migration_relational_engine import (
    get_migration_relational_engine,
)

from cognee.tasks.storage.index_data_points import index_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges

from uuid import uuid4

from cognee.modules.engine.models import TableRow, TableType

logger = logging.getLogger(__name__)


async def migrate_relational_database_kuzu(kuzu_adapter, schema):
    """
    Migrates data from a relational database into a Kuzu graph database.

    For each table in the schema:
      - Creates a TableType node representing the table
      - Fetches all rows and creates a TableRow node for each row
      - Links each TableRow node to its TableType node with an "is_part_of" relationship
    Then, for every foreign key defined in the schema:
      - Establishes relationships between TableRow nodes based on foreign key relationships

    Both TableType and TableRow inherit from DataPoint to maintain consistency with the
    application's data model.
    """
    engine = get_migration_relational_engine()
    nodes = []
    # Create a mapping of node_id to node objects for referencing in edge creation
    node_mapping = {}

    async with engine.engine.begin() as cursor:
        # First, create table type nodes for all tables
        for table_name, details in schema.items():
            # Create a TableType node for each table
            table_node = TableType(
                name=table_name, text=table_name, description=f"Table: {table_name}"
            )
            nodes.append(table_node)
            await kuzu_adapter.add_node(table_node)
            node_mapping[table_name] = table_node

            # Fetch all rows for the current table
            rows_result = await cursor.execute(text(f"SELECT * FROM {table_name};"))
            rows = rows_result.fetchall()

            for row in rows:
                # Build a dictionary of properties from the row
                row_properties = {
                    col["name"]: row[idx] for idx, col in enumerate(details["columns"])
                }

                # Determine the primary key value
                if not details["primary_key"]:
                    # Use the first column as primary key if not specified
                    primary_key_col = details["columns"][0]["name"]
                    primary_key_value = row_properties[primary_key_col]
                else:
                    # Use value of the specified primary key column
                    primary_key_col = details["primary_key"]
                    primary_key_value = row_properties[primary_key_col]

                # Create a node ID in the format "table_name:primary_key_value"
                node_id = f"{table_name}:{primary_key_value}"

                # Create a TableRow node
                row_node = TableRow(
                    name=node_id,
                    text=node_id,
                    properties=str(row_properties),
                    description=f"Row in {table_name} with {primary_key_col}={primary_key_value}",
                )
                nodes.append(row_node)

                # Add the row node to the graph
                await kuzu_adapter.add_node(row_node)

                # Store the node object in our mapping
                node_mapping[node_id] = row_node

                # Create edge between row node and table node
                await kuzu_adapter.add_edge(
                    from_node=row_node.id,
                    to_node=table_node.id,
                    relationship_name="is_part_of",
                )

        # Process foreign key relationships after all nodes are created
        for table_name, details in schema.items():
            # Process foreign key relationships for the current table
            for fk in details.get("foreign_keys", []):
                # Aliases needed for self-referencing tables
                alias_1 = f"{table_name}_e1"
                alias_2 = f"{fk['ref_table']}_e2"

                # Determine primary key column
                if not details["primary_key"]:
                    primary_key_col = details["columns"][0]["name"]
                else:
                    primary_key_col = details["primary_key"]

                # Query to find relationships based on foreign keys
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
                    # Construct node ids
                    source_node_id = f"{table_name}:{source_id}"
                    target_node_id = f"{fk['ref_table']}:{ref_value}"

                    # Check if both nodes exist in our mapping before creating edge
                    if source_node_id in node_mapping and target_node_id in node_mapping:
                        # Get the source and target node objects from our mapping
                        source_node = node_mapping[source_node_id]
                        target_node = node_mapping[target_node_id]

                        # Add edge representing the foreign key relationship using the node objects
                        await kuzu_adapter.add_edge(
                            from_node=source_node.id,
                            to_node=target_node.id,
                            relationship_name=fk["column"],
                            edge_properties={"relationship_type": fk["column"]},
                        )

    await index_data_points(nodes)
    # This step has to happen after adding nodes and edges because we query the graph.
    await index_graph_edges()

    logger.info("Data successfully migrated from relational database to Kuzu graph database")
    return await kuzu_adapter.get_graph_data()


async def migrate_relational_database_neo4j(graph_db, schema):
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

    async with engine.engine.begin() as connection:
        # Migrate rows as nodes and connect them to their table node.
        for table_name, details in schema.items():
            # Fetch all rows for the current table.
            rows_result = await connection.execute(text(f"SELECT * FROM {table_name};"))
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

                    # Add extra properties
                    properties["label"] = table_name
                    properties["type"] = "Table"

                    # Ensure the table node exists.
                    await session.run(
                        """
                        MERGE (t:Table {id: $table_name})
                        SET t += $properties
                        """,
                        table_name=table_name,
                        properties=properties,
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
                fk_result = await connection.execute(fk_query)
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


async def migrate_relational_database_networkx(networx_graph, schema):
    """
    Populates the NetworkX graph from a relational schema.

    For each table in the schema:
      - Fetch all rows and add each as a node. The node id is built as "<table_name>:<primary_key>"
      - For every foreign key defined, fetch the relationships and add an edge between the
        corresponding nodes if they exist.
    """
    engine = get_migration_relational_engine()

    async with engine.engine.begin() as cursor:
        # Iterate over all tables defined in the schema.
        # Migrate all rows from tables to graph
        for table_name, details in schema.items():
            # Fetch all rows for the current table.
            rows_result = await cursor.execute(text(f"SELECT * FROM {table_name};"))
            rows = rows_result.fetchall()

            for row in rows:
                # Build a dictionary of properties from the row.
                properties = {col["name"]: row[idx] for idx, col in enumerate(details["columns"])}

                if not details["primary_key"]:
                    # Assume the value of the first column in details['columns'] is the primary key.
                    node_id = f"{table_name}:{properties[details['columns'][0]['name']]}"
                else:
                    # Use value of the primary key column
                    node_id = f"{table_name}:{properties[details['primary_key']]}"

                # Also store the table name (or label) in the node attributes.
                properties["label"] = table_name
                properties["type"] = "TableRow"
                # Add the node to the graph.
                networx_graph.graph.add_node(node_id, **properties)

                # Add table node if it doesn't exist
                networx_graph.graph.add_node(table_name, type="Table")
                # Create edge between table and table element
                await networx_graph.add_edge(node_id, table_name, "is_part_of")

        # Iterate over all tables defined in the schema.
        # Map relationships between rows (which are now nodes in the graph) as edges in graph
        # NOTE: First all rows must be migrated to graph as nodes
        for table_name, details in schema.items():
            # Process foreign key relationships for the current table.
            for fk in details.get("foreign_keys", []):
                # Aliases are needed in the case a table is referencing itself
                alias_1 = f"{table_name}_e1"
                alias_2 = f"{fk['ref_table']}_e2"

                if not details["primary_key"]:
                    # Assume the first column in details['columns'] is the primary key.
                    primary_key_col = details["columns"][0]["name"]
                else:
                    primary_key_col = details["primary_key"]

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
                    # Construct node ids using the primary key value for the source.
                    source_node = f"{table_name}:{source_id}"
                    target_node = f"{fk['ref_table']}:{ref_value}"

                    if source_node in networx_graph.graph and target_node in networx_graph.graph:
                        networx_graph.graph.add_edge(
                            source_node,
                            target_node,
                            key=fk["column"],
                            relationship_type=fk["column"],
                        )

    # Save the updated graph to file.
    await networx_graph.save_graph_to_file(networx_graph.filename)
    print("Data populated into NetworkX successfully.")
