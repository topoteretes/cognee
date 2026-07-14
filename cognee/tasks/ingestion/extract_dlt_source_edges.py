import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid5, NAMESPACE_OID

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.databases.provenance import graph_provenance_write_kwargs
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.tasks.schema.models import SchemaTable, SchemaRelationship
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("extract_dlt_source_edges")


def _get_source_document(data_point: DataPoint):
    """Return the DltSourceDocument a data point belongs to, or None."""
    doc = getattr(data_point, "is_part_of", None)
    if doc is None:
        return None
    from cognee.modules.data.processing.document_types import DltSourceDocument

    return doc if isinstance(doc, DltSourceDocument) else None


async def extract_dlt_source_edges(
    data_points: List[DataPoint],
    ctx: Optional["PipelineContext"] = None,
) -> List[DataPoint]:
    """Create graph edges and schema nodes from a DLT source manifest.

    This task runs after add_data_points in the DLT cognify pipeline. The
    incoming data points are per-row DocumentChunks whose ids are the stable
    row node ids from the manifest. It:
    1. Creates SchemaTable nodes for each source table
    2. Creates SchemaRelationship nodes for each foreign key definition
    3. Creates is_row_of edges from row chunks to their SchemaTable
    4. Creates FK-based edges between row chunks of related rows

    Schema nodes use deterministic uuid5 ids, so re-emitting them across
    batches is an idempotent upsert. Row-level edges are only emitted for
    rows present in the current batch, so batching does not duplicate work.

    On a ledger graph the writes are registered in the relational rollback
    ledger; on a graph-provenance graph they are stamped in-graph instead
    (mirrors extract_dlt_fk_edges — no dual tracking).
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    # Group row chunks in this batch by their source document.
    source_docs = {}  # doc_id -> document
    batch_rows_by_doc: dict[str, set] = {}  # doc_id -> {row node id (str), ...}
    for data_point in data_points:
        doc = _get_source_document(data_point)
        if doc is None:
            continue
        doc_id = str(doc.id)
        source_docs[doc_id] = doc
        batch_rows_by_doc.setdefault(doc_id, set()).add(str(data_point.id))

    if not source_docs:
        return data_points

    graph_engine = await get_graph_engine()
    provenance_kwargs = await graph_provenance_write_kwargs(graph_engine, ctx)

    schema_nodes = []
    schema_edges = []
    fk_row_edges = []
    seen_row_edges = set()
    table_node_ids = {}
    fk_defs_seen = set()  # (table, column, ref_table, ref_column) for dedup
    relationship_count = 0

    for doc_id, doc in source_docs.items():
        async with open_data_file(doc.raw_data_location, mode="r", encoding="utf-8") as file:
            manifest = json.loads(file.read())

        tables = manifest.get("tables", {})
        rows = manifest.get("rows", [])
        batch_row_ids = batch_rows_by_doc.get(doc_id, set())

        # Phase 1: SchemaTable nodes for each source table
        for table_name, table_meta in tables.items():
            if table_name in table_node_ids:
                continue

            schema_info = table_meta.get("schema_info")
            columns_str = (
                json.dumps(schema_info, default=str)
                if isinstance(schema_info, (list, dict))
                else "[]"
            )
            fk_str = json.dumps(table_meta.get("foreign_keys", []), default=str)

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
                    f"from database '{table_meta.get('dlt_db_name', '')}'."
                ),
            )
            schema_nodes.append(table_node)
            table_node_ids[table_name] = table_node.id

        # Phase 2: SchemaRelationship nodes for each FK definition
        for table_name, table_meta in tables.items():
            for fk in table_meta.get("foreign_keys", []):
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
                schema_nodes.append(relationship)
                relationship_count += 1

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

        # Phase 3: row-level edges, limited to rows in the current batch
        for row in rows:
            row_node_id = row.get("node_id", "")
            if row_node_id not in batch_row_ids:
                continue

            table_name = row.get("table_name", "")
            table_node_id = table_node_ids.get(table_name)
            if table_node_id:
                schema_edges.append(
                    (
                        row_node_id,
                        str(table_node_id),
                        "is_row_of",
                        {
                            "source_node_id": row_node_id,
                            "target_node_id": str(table_node_id),
                            "relationship_name": "is_row_of",
                        },
                    )
                )

            for ref in row.get("fk_references", []):
                target_node_id = ref.get("target_data_id")
                relationship_name = ref.get("relationship_name", "references")

                if not target_node_id:
                    continue

                edge_key = (row_node_id, target_node_id, relationship_name)
                if edge_key in seen_row_edges:
                    continue
                seen_row_edges.add(edge_key)

                fk_row_edges.append(
                    (
                        row_node_id,
                        target_node_id,
                        relationship_name,
                        {
                            "source_node_id": row_node_id,
                            "target_node_id": target_node_id,
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
        await graph_engine.add_nodes(schema_nodes, **provenance_kwargs)
        await index_data_points(schema_nodes)
        logger.info(
            "Added %d schema nodes to graph (%d tables, %d relationships).",
            len(schema_nodes),
            len(table_node_ids),
            relationship_count,
        )

    all_edges = schema_edges + fk_row_edges
    if all_edges:
        await graph_engine.add_edges(all_edges, **provenance_kwargs)
        logger.info(
            "Added %d edges to graph (%d schema edges, %d FK row edges).",
            len(all_edges),
            len(schema_edges),
            len(fk_row_edges),
        )

    # Register writes in the relational rollback ledger unless they were
    # already stamped in-graph (mirrors extract_dlt_fk_edges).
    stamped_in_graph = provenance_kwargs["source_ref_key"] is not None
    if (
        not stamped_in_graph
        and (schema_nodes or all_edges)
        and ctx is not None
        and getattr(ctx, "user", None) is not None
        and getattr(ctx, "dataset", None) is not None
        and getattr(ctx, "data_item", None) is not None
        and getattr(ctx, "pipeline_run_id", None) is not None
    ):
        from cognee.modules.graph.methods import upsert_edges, upsert_nodes

        if schema_nodes:
            await upsert_nodes(
                schema_nodes,
                tenant_id=ctx.user.tenant_id,
                user_id=ctx.user.id,
                dataset_id=ctx.dataset.id,
                data_id=ctx.data_item.id,
                pipeline_run_id=ctx.pipeline_run_id,
            )
        if all_edges:
            await upsert_edges(
                all_edges,
                tenant_id=ctx.user.tenant_id,
                user_id=ctx.user.id,
                data_id=ctx.data_item.id,
                dataset_id=ctx.dataset.id,
                pipeline_run_id=ctx.pipeline_run_id,
            )

    return data_points
