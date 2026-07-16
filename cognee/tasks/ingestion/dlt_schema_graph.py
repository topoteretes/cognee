"""Shared emitter for DLT schema graph structure.

Builds and persists SchemaTable/SchemaRelationship nodes and row-level edges
for DLT-sourced relational data. Used by both extract_dlt_source_edges
(manifest pipeline) and extract_dlt_fk_edges (legacy per-row pipeline).
"""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from uuid import uuid5, NAMESPACE_OID

from cognee.infrastructure.databases.provenance import graph_provenance_write_kwargs
from cognee.tasks.schema.models import SchemaTable, SchemaRelationship
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("dlt_schema_graph")


def _table_node_id(table_name: str):
    """Deterministic node id for a DLT SchemaTable, shared by nodes and edges."""
    return uuid5(NAMESPACE_OID, name=f"dlt:{table_name}")


async def emit_dlt_schema_graph(
    tables: dict,
    row_records: list[dict],
    ctx: Optional["PipelineContext"] = None,
) -> None:
    """Build and persist the DLT schema graph for the given tables and rows.

    Args:
        tables: {table_name: {"schema_info", "foreign_keys", "dlt_db_name"}}
        row_records: [{"source_id": str, "table_name": str, "fk_references": [...]}]
        ctx: optional pipeline context for provenance/ledger registration.

    Creates SchemaTable nodes (deterministic uuid5 ids, so re-emitting is an
    idempotent upsert), SchemaRelationship nodes with has_foreign_key /
    references_table edges, is_row_of edges from each row to its table node,
    and FK-based edges between rows from fk_references.

    On a ledger graph the writes are registered in the relational rollback
    ledger; on a graph-provenance graph they are stamped in-graph instead
    (mirrors add_data_points — no dual tracking).
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if not tables and not row_records:
        return

    graph_engine = await get_graph_engine()
    provenance_kwargs = await graph_provenance_write_kwargs(graph_engine, ctx)

    schema_nodes = []
    schema_edges = []
    fk_row_edges = []
    seen_row_edges = set()
    fk_defs_seen = set()  # (table, column, ref_table, ref_column) for dedup

    # SchemaTable nodes for each source table
    table_node_ids = {}
    for table_name, table_meta in tables.items():
        schema_info = table_meta["schema_info"]
        columns_str = (
            json.dumps(schema_info, default=str) if isinstance(schema_info, (list, dict)) else "[]"
        )
        fk_str = json.dumps(table_meta["foreign_keys"], default=str)

        table_node = SchemaTable(
            id=_table_node_id(table_name),
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

    # SchemaRelationship nodes for each FK definition
    relationship_count = 0
    for table_name, table_meta in tables.items():
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

    # Row-level edges: is_row_of to the table node, plus FK edges between rows
    for record in row_records:
        source_id = record["source_id"]
        table_name = record.get("table_name", "")

        # Table node ids are deterministic, so the edge target is valid even
        # when the table node was emitted by an earlier batch (tables omitted).
        if table_name:
            schema_edges.append(
                (
                    source_id,
                    str(_table_node_id(table_name)),
                    "is_row_of",
                    {
                        "source_node_id": source_id,
                        "target_node_id": str(_table_node_id(table_name)),
                        "relationship_name": "is_row_of",
                    },
                )
            )

        for ref in record.get("fk_references", []):
            # Legacy metadata uses "target_data_id"; manifests use "target_node_id".
            target_node_id = ref.get("target_node_id") or ref.get("target_data_id")
            relationship_name = ref.get("relationship_name", "references")

            if not target_node_id:
                continue

            edge_key = (source_id, target_node_id, relationship_name)
            if edge_key in seen_row_edges:
                continue
            seen_row_edges.add(edge_key)

            fk_row_edges.append(
                (
                    source_id,
                    target_node_id,
                    relationship_name,
                    {
                        "source_node_id": source_id,
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

    # Register the schema nodes and FK edges in the relational rollback ledger
    # so the cognify rollback handler can find and remove them on failure or
    # startup recovery. When the writes were stamped in-graph, provenance lives
    # in the graph and the ledger is skipped (mirrors add_data_points — no dual
    # tracking). Skipped when no pipeline context is available (e.g. direct
    # task invocation).
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
