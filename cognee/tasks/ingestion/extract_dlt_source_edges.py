from typing import TYPE_CHECKING, Optional

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.dlt_schema_graph import emit_dlt_schema_graph
from cognee.tasks.ingestion.dlt_utils import load_dlt_manifest

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
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
) -> list[DataPoint]:
    """Create graph edges and schema nodes from a DLT source manifest.

    This task runs after add_data_points in the DLT cognify pipeline. The
    incoming data points are per-row DocumentChunks whose ids are the stable
    row node ids from the manifest. It gathers table schemas and per-row FK
    records from the manifest and delegates graph construction to
    emit_dlt_schema_graph.

    Row-level edges are only emitted for rows present in the current batch,
    so batching does not duplicate work. An FK edge waits for the batch that
    writes the later of its two rows, since the graph drops an edge whose
    target row is not written yet. Cross-batch state lives in
    ``ctx.extras``: the rows written so far, a doc's schema nodes are emitted
    (and vector-embedded) only for the first batch that carries its rows, and
    each unique ColumnValue node is persisted and embedded once. ``ctx`` is per data
    item, and one manifest data item IS one source — that equality is what
    makes ctx.extras the correct scope. If manifests ever stop being
    one-item-per-source, this state needs a new home.
    """
    if ctx is not None:
        emitted_schema_docs = ctx.extras.setdefault("dlt_emitted_schema_docs", set())
        emitted_value_node_ids = ctx.extras.setdefault("dlt_emitted_value_node_ids", set())
        written_row_ids = ctx.extras.setdefault("dlt_written_row_ids", set())
    else:
        # Direct invocation (unit tests, custom pipelines) without a pipeline
        # context: fresh per-call sets — batching dedup only.
        emitted_schema_docs = set()
        emitted_value_node_ids = set()
        written_row_ids = set()
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

    tables: dict[str, dict] = {}
    row_records: list[dict] = []
    newly_emitted_doc_ids: list[str] = []

    for doc_id, doc in source_docs.items():
        manifest = await load_dlt_manifest(doc.raw_data_location)

        # Emit a doc's schema nodes only once per data item's run. A table's key
        # and size come from its rows, which every manifest carries. Table nodes
        # are scoped to their source document: another source's table of the same
        # name is a different table.
        if doc_id not in emitted_schema_docs:
            rows_of: dict[str, list[dict]] = {}
            for row in manifest.get("rows", []):
                rows_of.setdefault(row["table_name"], []).append(row)
            for table_name, table_meta in manifest.get("tables", {}).items():
                tables[f"{doc_id}:{table_name}"] = {
                    "name": table_name,
                    "scope": doc_id,
                    "source_label": manifest.get("source_label"),
                    "schema_info": table_meta.get("schema_info"),
                    "foreign_keys": table_meta.get("foreign_keys", []),
                    "primary_key": rows_of[table_name][0]["primary_key_column"],
                    "row_count": len(rows_of[table_name]),
                }
            newly_emitted_doc_ids.append(doc_id)

        # Row-level records for the rows in the current batch, plus the FK edges
        # of rows written by earlier batches whose target rows are in this one.
        # add_data_points has already written this batch's rows.
        batch_row_ids = batch_rows_by_doc[doc_id]
        rows_in_graph = written_row_ids | batch_row_ids
        for row in manifest.get("rows", []):
            row_node_id = row["node_id"]
            fk_references = row.get("fk_references", [])
            if row_node_id in batch_row_ids:
                row_records.append(
                    {
                        "source_id": row_node_id,
                        "table_name": row.get("table_name", ""),
                        "scope": doc_id,
                        "fk_references": [
                            ref for ref in fk_references if ref["target_node_id"] in rows_in_graph
                        ],
                        "column_values": row.get("column_values") or {},
                        "timestamps": row.get("timestamps") or {},
                    }
                )
            elif row_node_id in written_row_ids:
                arrived_targets = [
                    ref for ref in fk_references if ref["target_node_id"] in batch_row_ids
                ]
                if arrived_targets:
                    row_records.append(
                        {
                            "source_id": row_node_id,
                            "table_name": row.get("table_name", ""),
                            "scope": doc_id,
                            "fk_references": arrived_targets,
                            "fk_only": True,
                        }
                    )

    await emit_dlt_schema_graph(
        tables,
        row_records,
        ctx,
        emitted_value_node_ids=emitted_value_node_ids,
    )

    emitted_schema_docs.update(newly_emitted_doc_ids)
    for batch_row_ids in batch_rows_by_doc.values():
        written_row_ids.update(batch_row_ids)

    return data_points
