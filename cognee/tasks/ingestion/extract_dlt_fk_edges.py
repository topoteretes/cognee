"""Legacy: handles pre-manifest per-row DLT data only; remove once migrated."""

from typing import TYPE_CHECKING, List, Optional

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.dlt_schema_graph import emit_dlt_schema_graph
from cognee.tasks.ingestion.dlt_utils import parse_external_metadata

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("extract_dlt_fk_edges")


def _is_dlt_data_point(data_point: DataPoint) -> bool:
    """Fast check whether a data point originates from a DLT source."""
    doc = getattr(data_point, "is_part_of", None)
    if doc is None:
        return False
    from cognee.modules.data.processing.document_types import DltRowDocument

    return isinstance(doc, DltRowDocument)


async def extract_dlt_fk_edges(
    data_points: List[DataPoint],
    ctx: Optional["PipelineContext"] = None,
) -> List[DataPoint]:
    """Create graph edges and schema nodes from legacy per-row DLT data.

    This task runs after add_data_points in the cognify pipeline. It gathers
    table schemas and per-row FK references from the documents'
    external_metadata and delegates graph construction to
    emit_dlt_schema_graph (shared with extract_dlt_source_edges).
    """
    # Quick check: skip entirely if no data points have DLT metadata.
    # This avoids iterating + parsing metadata for pure unstructured batches.
    if not any(_is_dlt_data_point(dp) for dp in data_points):
        return data_points

    # Collect tables and per-row FK records from all documents in this batch
    tables: dict[str, dict] = {}
    row_records: list[dict] = []
    seen_doc_ids: set[str] = set()

    for data_point in data_points:
        doc = getattr(data_point, "is_part_of", None)
        if doc is None or str(doc.id) in seen_doc_ids:
            continue

        ext_metadata = parse_external_metadata(doc)
        if ext_metadata is None or ext_metadata.get("source") != "dlt":
            continue
        seen_doc_ids.add(str(doc.id))

        table_name = ext_metadata.get("table_name", "")
        if table_name and table_name not in tables:
            tables[table_name] = {
                "schema_info": ext_metadata.get("schema_info"),
                "foreign_keys": ext_metadata.get("foreign_keys", []),
                "dlt_db_name": ext_metadata.get("dlt_db_name", ""),
            }

        row_records.append(
            {
                "source_id": str(doc.id),
                "table_name": table_name,
                "fk_references": ext_metadata.get("fk_references", []),
            }
        )

    await emit_dlt_schema_graph(tables, row_records, ctx)

    return data_points
