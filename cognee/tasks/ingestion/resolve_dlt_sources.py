"""DLT adapter: expand DLT resources into standard DataItem objects.

Called *before* the add pipeline so that ingest_data never sees DLT types.
One-to-many expansion (one DLT source → many rows) happens here; the
per-item pipeline model downstream stays unchanged.
"""

from typing import Any, List, Optional, Set
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
        from dlt.extract.source import DltSource
    except ImportError:
        # dlt not installed — nothing to resolve
        return data

    primary_key = kwargs["primary_key"] if "primary_key" in kwargs else None
    write_disposition = kwargs["write_disposition"] if "write_disposition" in kwargs else "replace"
    query = kwargs["query"] if "query" in kwargs else None
    max_rows_per_table = kwargs.get("max_rows_per_table")

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
        if isinstance(item, (DltResource, DltSource, SourceFactory)):
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
            max_rows_per_table=max_rows_per_table,
        )
        all_rows.extend(rows)

    # --- Phase 1: compute stable data_ids for all rows (for FK resolution) --
    # Primary lookup uses content_hash for uniqueness (handles tables with
    # non-unique fallback PKs like junction tables).
    row_id_lookup: dict[tuple[str, str, str], UUID] = {}
    # FK lookup maps (table, pk_value) → data_id for foreign key resolution.
    # When multiple rows share a PK value, the last one wins (best-effort).
    fk_lookup: dict[tuple[str, str], UUID] = {}
    for row in all_rows:
        row_identifier = f"dlt:{row.table_name}:{row.primary_key_value}:{row.content_hash}"
        data_id = await get_unique_data_id(row_identifier, user)
        row_id_lookup[(row.table_name, row.primary_key_value, row.content_hash)] = data_id
        fk_lookup[(row.table_name, row.primary_key_value)] = data_id

    # --- Phase 2: create DataItems ------------------------------------------
    # Build table-level metadata once per table so all rows share the same
    # schema_info/foreign_keys references instead of duplicating per row.
    _table_meta_cache: dict[str, dict] = {}

    def _get_table_meta(row: DltRowData) -> dict:
        if row.table_name not in _table_meta_cache:
            _table_meta_cache[row.table_name] = {
                "schema_info": row.schema_info,
                "schema_hash": row.schema_hash,
                "foreign_keys": row.foreign_keys,
                "dlt_db_name": row.dlt_db_name,
            }
        return _table_meta_cache[row.table_name]

    expanded_items: list[DataItem] = []
    for row in all_rows:
        data_id = row_id_lookup[(row.table_name, row.primary_key_value, row.content_hash)]

        enriched_text = _build_schema_context_text(row)
        fk_references = _resolve_fk_references(row, fk_lookup)
        table_meta = _get_table_meta(row)

        ext_metadata = {
            "source": "dlt",
            "table_name": row.table_name,
            "primary_key_column": row.primary_key_column,
            "primary_key_value": row.primary_key_value,
            "schema_info": table_meta["schema_info"],
            "schema_hash": table_meta["schema_hash"],
            "foreign_keys": table_meta["foreign_keys"],
            "fk_references": fk_references,
            "dlt_db_name": table_meta["dlt_db_name"],
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

    # --- Phase 3: delete orphaned dlt rows no longer in the source ----------
    # Skip orphan deletion for "append" disposition — each run intentionally
    # adds new rows, so prior batches should not be treated as orphans.
    if write_disposition != "append":
        fresh_data_ids: Set[UUID] = set(row_id_lookup.values())
        await _delete_dlt_orphans(dataset_name, user, fresh_data_ids)

    result = non_dlt_items + expanded_items
    return result


# ---------------------------------------------------------------------------
# Helpers (moved from ingest_data.py)
# ---------------------------------------------------------------------------


def _build_schema_context_text(dlt_row: DltRowData) -> str:
    """Build a human-readable, schema-enriched text representation of a DLT row.

    This text is stored as the document content and used for vector search.
    DLT rows bypass LLM extraction — their graph is built deterministically
    from the relational schema by ``extract_dlt_fk_edges``.
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


async def _delete_dlt_orphans(
    dataset_name: str,
    user: User,
    fresh_data_ids: Set[UUID],
) -> None:
    """Delete dlt-sourced Data records (and their graph/vector artifacts) that
    are no longer present in the freshly-ingested dlt source.

    This handles the case where rows are deleted from the upstream database
    and the user re-ingests.  dlt cleans its own staging DB, but cognee's
    relational, graph, and vector stores still hold stale data.
    """
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.data.methods.delete_data import delete_data
    from cognee.modules.graph.methods.has_data_related_nodes import has_data_related_nodes
    from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
        delete_data_nodes_and_edges,
    )

    # Find the dataset — if it doesn't exist yet this is a first ingestion,
    # so there can be no orphans.
    existing_datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[dataset_name]
    )
    if not existing_datasets:
        return

    dataset = existing_datasets[0]
    all_data: list = await get_dataset_data(dataset.id)

    orphans = []
    for data_item in all_data:
        ext = data_item.external_metadata
        if not isinstance(ext, dict) or ext.get("source") != "dlt":
            continue
        if data_item.id not in fresh_data_ids:
            orphans.append(data_item)

    if not orphans:
        return

    logger.info(
        "Deleting %d orphaned dlt row(s) from dataset '%s'.",
        len(orphans),
        dataset_name,
    )

    for orphan in orphans:
        try:
            if await has_data_related_nodes(dataset.id, orphan.id):
                await delete_data_nodes_and_edges(dataset.id, orphan.id, user.id)
            await delete_data(orphan, dataset.id)
        except Exception:
            logger.warning(
                "Failed to delete orphaned dlt row data_id=%s, skipping.",
                orphan.id,
                exc_info=True,
            )
