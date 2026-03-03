"""DLT adapter: expand DLT resources into standard DataItem objects.

Called *before* the add pipeline so that ingest_data never sees DLT types.
One-to-many expansion (one DLT source → many rows) happens here; the
per-item pipeline model downstream stays unchanged.
"""

from typing import Any, List, Optional
from uuid import UUID

from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

from .create_dlt_source import (
    is_connection_string,
    is_csv_path,
    create_dlt_source_from_connection_string,
    create_dlt_source_from_csv,
)
from .data_item import DataItem
from .dlt_row_data import DltRowData
from .ingest_dlt_source import ingest_dlt_source

logger = get_logger("resolve_dlt_sources")


async def resolve_dlt_sources(
    data: Any,
    dataset_name: str,
    user: User,
    **kwargs,
) -> Any:
    """Expand DLT resources (and auto-detected structured data) into DataItems.

    Non-DLT items pass through unchanged.  DLT resources are ingested via
    ``ingest_dlt_source`` and each resulting row is wrapped in a ``DataItem``
    with stable ``data_id``, enriched text, and ``external_metadata``.

    Returns the (possibly expanded) data — either a single item, a list, or
    unchanged if nothing was DLT.
    """
    # Lazy-import DLT types so the dlt package is not a hard dependency
    try:
        from dlt.extract import DltResource, SourceFactory
    except ImportError:
        # dlt not installed — nothing to resolve
        return data

    primary_key = kwargs["primary_key"] if "primary_key" in kwargs else None
    write_disposition = kwargs["write_disposition"] if "write_disposition" in kwargs else None
    query = kwargs["query"] if "query" in kwargs else None

    # --- Auto-detect structured data (CSV paths / connection strings) ------
    if isinstance(data, str):
        if is_csv_path(data):
            data = create_dlt_source_from_csv(data)
        elif is_connection_string(data):
            data = create_dlt_source_from_connection_string(data, query=query)

    # Normalise to list for uniform processing
    data_list = data if isinstance(data, list) else [data]

    dlt_items = []
    non_dlt_items = []

    for item in data_list:
        if isinstance(item, (DltResource, SourceFactory)):
            dlt_items.append(item)
        else:
            non_dlt_items.append(item)

    if not dlt_items:
        # Nothing to expand — return original data unchanged
        return data

    # --- Run DLT pipelines and collect rows ---------------------------------
    all_rows: List[DltRowData] = []
    for dlt_item in dlt_items:
        rows = await ingest_dlt_source(
            dlt_item,
            dataset_name,
            primary_key=primary_key,
            write_disposition=write_disposition,
        )
        all_rows.extend(rows)

    # --- Phase 1: compute stable data_ids for all rows (for FK resolution) --
    row_id_lookup: dict[tuple[str, str], UUID] = {}
    for row in all_rows:
        row_identifier = f"dlt:{row.table_name}:{row.primary_key_value}"
        data_id = await get_unique_data_id(row_identifier, user)
        row_id_lookup[(row.table_name, row.primary_key_value)] = data_id

    # --- Phase 2: create DataItems ------------------------------------------
    expanded_items: list[DataItem] = []
    for row in all_rows:
        data_id = row_id_lookup[(row.table_name, row.primary_key_value)]

        enriched_text = _build_schema_context_text(row)
        fk_references = _resolve_fk_references(row, row_id_lookup)

        ext_metadata = {
            "source": "dlt",
            "table_name": row.table_name,
            "primary_key_column": row.primary_key_column,
            "primary_key_value": row.primary_key_value,
            "schema_info": row.schema_info,
            "schema_hash": row.schema_hash,
            "foreign_keys": row.foreign_keys,
            "fk_references": fk_references,
            "dlt_db_name": row.dlt_db_name,
            "content_hash": row.content_hash,
        }

        item = DataItem(
            data=enriched_text,
            label=f"{row.table_name}:{row.primary_key_value}",
            external_metadata=ext_metadata,
            data_id=data_id,
        )
        expanded_items.append(item)

    logger.info("Resolved %d DLT source(s) into %d DataItems.", len(dlt_items), len(expanded_items))

    result = non_dlt_items + expanded_items
    return result


# ---------------------------------------------------------------------------
# Helpers (moved from ingest_data.py)
# ---------------------------------------------------------------------------


def _build_schema_context_text(dlt_row: DltRowData) -> str:
    """Build a schema-enriched text representation of a DLT row.

    Instead of raw JSON, this gives the LLM structural context about the table,
    column types, and foreign key relationships so it can extract better entities.
    """
    lines = []
    lines.append(f"Table: {dlt_row.table_name}")

    # Build column descriptions
    schema_info = dlt_row.schema_info
    col_descriptions = []
    if isinstance(schema_info, list):
        # SQLite format: [{"name": "col", "type": "TEXT"}, ...]
        for col in schema_info:
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            col_descriptions.append(f"  - {col_name} ({col_type})")
    elif isinstance(schema_info, dict):
        # Postgres format: {"col_name": {"type": "TEXT", ...}, ...}
        for col_name, col_info in schema_info.items():
            col_type = col_info.get("type", "") if isinstance(col_info, dict) else str(col_info)
            col_descriptions.append(f"  - {col_name} ({col_type})")

    if col_descriptions:
        lines.append("Columns:")
        lines.extend(col_descriptions)

    # Add FK context
    if dlt_row.foreign_keys:
        fk_lines = []
        for fk in dlt_row.foreign_keys:
            col = fk.get("column", "")
            ref_table = fk.get("ref_table", "")
            ref_col = fk.get("ref_column", "")
            fk_lines.append(f"  - {col} references {ref_table}.{ref_col}")
        if fk_lines:
            lines.append("Foreign Keys:")
            lines.extend(fk_lines)

    # Add the actual row data
    lines.append("")
    lines.append("Row Data:")
    for key, value in dlt_row.row_data.items():
        lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def _resolve_fk_references(dlt_row: DltRowData, row_id_lookup: dict) -> list:
    """Resolve foreign key columns to target data_ids for graph edge creation.

    Returns a list of dicts:
    [{"column": "dept_id", "target_table": "departments", "target_pk_value": "10",
      "target_data_id": "uuid-string", "relationship_name": "dept_id_references_departments"}]
    """
    references = []
    for fk in dlt_row.foreign_keys:
        fk_column = fk.get("column", "")
        ref_table = fk.get("ref_table", "")
        ref_column = fk.get("ref_column", "")

        if not fk_column or not ref_table:
            continue

        # Get the FK value from the row data
        fk_value = dlt_row.row_data.get(fk_column)
        if fk_value is None:
            continue

        fk_value_str = str(fk_value)
        target_key = (ref_table, fk_value_str)
        target_data_id = row_id_lookup.get(target_key)

        if target_data_id is not None:
            relationship_name = f"{fk_column}_references_{ref_table}"
            references.append(
                {
                    "column": fk_column,
                    "target_table": ref_table,
                    "target_column": ref_column,
                    "target_pk_value": fk_value_str,
                    "target_data_id": str(target_data_id),
                    "relationship_name": relationship_name,
                }
            )

    return references
