import logging
from uuid import uuid5, NAMESPACE_OID
from sqlalchemy import text
from cognee.infrastructure.databases.relational.get_migration_relational_engine import (
    get_migration_relational_engine,
)

from cognee.tasks.storage.index_data_points import index_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges

from cognee.modules.engine.models import TableRow, TableType

logger = logging.getLogger(__name__)


async def migrate_relational_database(graph_db, schema):
    """
    Migrates data from a relational database into a graph database.

    For each table in the schema:
      - Creates a TableType node representing the table
      - Fetches all rows and creates a TableRow node for each row
      - Links each TableRow node to its TableType node with an "is_part_of" relationship
    Then, for every foreign key defined in the schema:
      - Establishes relationships between TableRow nodes based on foreign key relationships

    Both TableType and TableRow inherit from DataPoint to maintain consistency with Cognee data model.
    """
    engine = get_migration_relational_engine()
    # Create a mapping of node_id to node objects for referencing in edge creation
    node_mapping = {}
    edge_mapping = []

    async with engine.engine.begin() as cursor:
        # First, create table type nodes for all tables
        for table_name, details in schema.items():
            # Create a TableType node for each table
            table_node = TableType(
                id=uuid5(NAMESPACE_OID, name=table_name),
                name=table_name,
                description=f"Table: {table_name}",
            )

            # Add TableType node to mapping ( node will be added to the graph later based on this mapping )
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
                # Node id must uniquely map to the id used in the relational database
                # To catch the foreign key relationships properly
                row_node = TableRow(
                    id=uuid5(NAMESPACE_OID, name=node_id),
                    name=node_id,
                    is_a=table_node,
                    properties=str(row_properties),
                    description=f"Row in {table_name} with {primary_key_col}={primary_key_value}",
                )

                # Store the node object in our mapping
                node_mapping[node_id] = row_node

                # Add edge between row node and table node ( it will be added to the graph later )
                edge_mapping.append(
                    (
                        row_node.id,
                        table_node.id,
                        "is_part_of",
                        dict(
                            relationship_name="is_part_of",
                            source_node_id=row_node.id,
                            target_node_id=table_node.id,
                        ),
                    )
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

                    # Get the source and target node objects from our mapping
                    source_node = node_mapping[source_node_id]
                    target_node = node_mapping[target_node_id]

                    # Add edge representing the foreign key relationship using the node objects
                    # Create edge to add to graph later
                    edge_mapping.append(
                        (
                            source_node.id,
                            target_node.id,
                            fk["column"],
                            dict(
                                source_node_id=source_node.id,
                                target_node_id=target_node.id,
                                relationship_name=fk["column"],
                            ),
                        )
                    )

    def _remove_duplicate_edges(edge_mapping):
        seen = set()
        unique_original_shape = []

        for tup in edge_mapping:
            # We go through all the tuples in the edge_mapping and we only add unique tuples to the list
            # To eliminate duplicate edges.
            source_id, target_id, rel_name, rel_dict = tup
            # We need to convert the dictionary to a frozenset to be able to compare values for it
            rel_dict_hashable = frozenset(sorted(rel_dict.items()))
            hashable_tup = (source_id, target_id, rel_name, rel_dict_hashable)

            # We use the seen set to keep track of unique edges
            if hashable_tup not in seen:
                # A list that has frozensets elements instead of dictionaries is needed to be able to compare values
                seen.add(hashable_tup)
                # append the original tuple shape (with the dictionary) if it's the first time we see it
                unique_original_shape.append(tup)

        return unique_original_shape

    # Add all nodes and edges to the graph
    # NOTE: Nodes and edges have to be added in batch for speed optimization, Especially for NetworkX.
    #       If we'd create nodes and add them to graph in real time the process would take too long.
    #       Every node and edge added to NetworkX is saved to file which is very slow when not done in batches.
    await graph_db.add_nodes(list(node_mapping.values()))
    await graph_db.add_edges(_remove_duplicate_edges(edge_mapping))

    # In these steps we calculate the vector embeddings of our nodes and edges and save them to vector database
    # Cognee uses this information to perform searches on the knowledge graph.
    await index_data_points(list(node_mapping.values()))
    await index_graph_edges()

    logger.info("Data successfully migrated from relational database to desired graph database.")
    return await graph_db.get_graph_data()
