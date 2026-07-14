from typing import TYPE_CHECKING, List, Optional

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
    data_points: List[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    emitted_schema_docs: Optional[set] = None,
) -> List[DataPoint]:
    """Create graph edges and schema nodes from a DLT source manifest.

    This task runs after add_data_points in the DLT cognify pipeline. The
    incoming data points are per-row DocumentChunks whose ids are the stable
    row node ids from the manifest. It gathers table schemas and per-row FK
    records from the manifest and delegates graph construction to
    emit_dlt_schema_graph (shared with the legacy extract_dlt_fk_edges).

    Row-level edges are only emitted for rows present in the current batch,
    so batching does not duplicate work. ``emitted_schema_docs`` is shared
    across batches of one pipeline run via the Task kwarg: a doc's schema
    nodes are emitted (and vector-embedded) only for the first batch, avoiding
    re-embedding identical schema nodes per batch.
    """
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
        rows_by_node_id = {row["node_id"]: row for row in manifest.get("rows", [])}

        # Emit a doc's schema nodes only once per pipeline run.
        if emitted_schema_docs is None or doc_id not in emitted_schema_docs:
            for table_name, table_meta in manifest.get("tables", {}).items():
                if table_name in tables:
                    continue
                tables[table_name] = {
                    "schema_info": table_meta.get("schema_info"),
                    "foreign_keys": table_meta.get("foreign_keys", []),
                    "dlt_db_name": table_meta.get("dlt_db_name", ""),
                }
            newly_emitted_doc_ids.append(doc_id)

        # Row-level records, limited to rows in the current batch.
        for row_node_id in batch_rows_by_doc.get(doc_id, set()):
            row = rows_by_node_id.get(row_node_id)
            if row is None:
                continue
            row_records.append(
                {
                    "source_id": row_node_id,
                    "table_name": row.get("table_name", ""),
                    "fk_references": row.get("fk_references", []),
                }
            )

    await emit_dlt_schema_graph(tables, row_records, ctx)

    if emitted_schema_docs is not None:
        emitted_schema_docs.update(newly_emitted_doc_ids)

    return data_points
