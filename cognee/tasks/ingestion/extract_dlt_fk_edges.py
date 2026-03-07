import json
from datetime import datetime, timezone
from typing import List
from uuid import uuid5, NAMESPACE_OID

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.tasks.schema.models import SchemaTable, SchemaRelationship
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.dlt_utils import parse_external_metadata

logger = get_logger("extract_dlt_fk_edges")


def _is_dlt_data_point(data_point: DataPoint) -> bool:
    """Fast check whether a data point originates from a DLT source."""
    doc = getattr(data_point, "is_part_of", None)
    if doc is None:
        return False
    from cognee.modules.data.processing.document_types import DltRowDocument

    return isinstance(doc, DltRowDocument)


async def extract_dlt_fk_edges(data_points: List[DataPoint]) -> List[DataPoint]:
    """Create graph edges and schema nodes from DLT-sourced relational data.

    This task runs after add_data_points in the cognify pipeline. It:
    1. Creates SchemaTable nodes for each DLT source table (reuses existing models
       from cognee.tasks.schema)
    2. Creates SchemaRelationship nodes for each foreign key
    3. Creates FK-based edges between Document nodes of related rows

    This reuses the SchemaTable/SchemaRelationship models from the existing
    relational pipeline mapper (migrate_relational_database / ingest_database_schema).
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    # Quick check: skip entirely if no data points have DLT metadata.
    # This avoids iterating + parsing metadata for pure unstructured batches.
    has_any_dlt = any(_is_dlt_data_point(dp) for dp in data_points)
    if not has_any_dlt:
        return data_points

    # Collect DLT metadata from all documents in this batch
    dlt_docs = {}  # doc_id -> ext_metadata
    tables_seen = {}  # table_name -> schema_info
    fk_defs_seen = set()  # (table, column, ref_table, ref_column) for dedup

    for data_point in data_points:
        doc = getattr(data_point, "is_part_of", None)
        if doc is None:
            continue

        ext_metadata = parse_external_metadata(doc)
        if ext_metadata is None or ext_metadata.get("source") != "dlt":
            continue

        doc_id = str(doc.id)
        dlt_docs[doc_id] = ext_metadata

        table_name = ext_metadata.get("table_name", "")
        if table_name and table_name not in tables_seen:
            tables_seen[table_name] = {
                "schema_info": ext_metadata.get("schema_info"),
                "foreign_keys": ext_metadata.get("foreign_keys", []),
                "dlt_db_name": ext_metadata.get("dlt_db_name", ""),
            }

    if not dlt_docs:
        return data_points

    graph_engine = await get_graph_engine()
    schema_nodes = []
    schema_edges = []
    fk_row_edges = []
    seen_row_edges = set()

    # Phase 1: Create SchemaTable nodes for each source table
    table_node_ids = {}
    for table_name, table_meta in tables_seen.items():
        schema_info = table_meta["schema_info"]

        # Build column description
        if isinstance(schema_info, (list, dict)):
            columns_str = json.dumps(schema_info, default=str)
        else:
            columns_str = "[]"

        fk_str = json.dumps(table_meta["foreign_keys"], default=str)

        table_node = SchemaTable(
            id=uuid5(NAMESPACE_OID, name=f"dlt:{table_name}"),
            name=table_name,
            columns=columns_str,
            primary_key=None,
            foreign_keys=fk_str,
            sample_rows="[]",
            row_count_estimate=None,
            description=(
                f"DLT-ingested relational table '{table_name}' "
                f"from database '{table_meta['dlt_db_name']}'."
            ),
        )
        schema_nodes.append(table_node)
        table_node_ids[table_name] = table_node.id

    # Phase 2: Create SchemaRelationship nodes for each FK definition
    relationship_nodes = []
    for table_name, table_meta in tables_seen.items():
        for fk in table_meta["foreign_keys"]:
            fk_col = fk.get("column", "")
            ref_table = fk.get("ref_table", "")
            ref_col = fk.get("ref_column", "")

            if not fk_col or not ref_table:
                continue

            fk_key = (table_name, fk_col, ref_table, ref_col)
            if fk_key in fk_defs_seen:
                continue
            fk_defs_seen.add(fk_key)

            rel_name = f"{table_name}:{fk_col}->{ref_table}:{ref_col}"
            relationship = SchemaRelationship(
                id=uuid5(NAMESPACE_OID, name=f"dlt:{rel_name}"),
                name=rel_name,
                source_table=table_name,
                target_table=ref_table,
                relationship_type="foreign_key",
                source_column=fk_col,
                target_column=ref_col,
                description=(f"Foreign key: {table_name}.{fk_col} -> {ref_table}.{ref_col}"),
            )
            relationship_nodes.append(relationship)

            # Create edges: source_table -> relationship -> target_table
            source_table_id = table_node_ids.get(table_name)
            target_table_id = table_node_ids.get(ref_table)

            if source_table_id:
                schema_edges.append(
                    (
                        str(source_table_id),
                        str(relationship.id),
                        "has_foreign_key",
                        {
                            "source_node_id": str(source_table_id),
                            "target_node_id": str(relationship.id),
                            "relationship_name": "has_foreign_key",
                        },
                    )
                )
            if target_table_id:
                schema_edges.append(
                    (
                        str(relationship.id),
                        str(target_table_id),
                        "references_table",
                        {
                            "source_node_id": str(relationship.id),
                            "target_node_id": str(target_table_id),
                            "relationship_name": "references_table",
                        },
                    )
                )

    schema_nodes.extend(relationship_nodes)

    # Phase 3: Create row-level FK edges between Document nodes
    for doc_id, ext_metadata in dlt_docs.items():
        fk_references = ext_metadata.get("fk_references", [])
        table_name = ext_metadata.get("table_name", "")

        # Link each document to its SchemaTable
        table_node_id = table_node_ids.get(table_name)
        if table_node_id:
            schema_edges.append(
                (
                    doc_id,
                    str(table_node_id),
                    "is_row_of",
                    {
                        "source_node_id": doc_id,
                        "target_node_id": str(table_node_id),
                        "relationship_name": "is_row_of",
                    },
                )
            )

        for ref in fk_references:
            target_data_id = ref.get("target_data_id")
            relationship_name = ref.get("relationship_name", "references")

            if not target_data_id:
                continue

            edge_key = (doc_id, target_data_id, relationship_name)
            if edge_key in seen_row_edges:
                continue
            seen_row_edges.add(edge_key)

            fk_row_edges.append(
                (
                    doc_id,
                    target_data_id,
                    relationship_name,
                    {
                        "source_node_id": doc_id,
                        "target_node_id": target_data_id,
                        "relationship_name": relationship_name,
                        "edge_text": relationship_name.replace("_", " "),
                        "source_table": table_name,
                        "target_table": ref.get("target_table", ""),
                        "fk_column": ref.get("column", ""),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

    # Persist to graph
    if schema_nodes:
        await graph_engine.add_nodes(schema_nodes)
        await index_data_points(schema_nodes)
        logger.info(
            "Added %d schema nodes to graph (%d tables, %d relationships).",
            len(schema_nodes),
            len(table_node_ids),
            len(relationship_nodes),
        )

    all_edges = schema_edges + fk_row_edges
    if all_edges:
        await graph_engine.add_edges(all_edges)
        logger.info(
            "Added %d edges to graph (%d schema edges, %d FK row edges).",
            len(all_edges),
            len(schema_edges),
            len(fk_row_edges),
        )

    return data_points
