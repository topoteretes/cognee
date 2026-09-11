"""DLT adapter: resolve DLT resources into standard DataItem objects.

Called *before* the add pipeline so that ingest_data never sees DLT types.
A relational DLT source becomes a single manifest DataItem (one Data record
per source); the manifest is a JSON document describing every unique row,
materialized as individual graph/vector nodes later by the dedicated DLT
cognify pipeline (see extract_dlt_source_edges). A source may instead opt
into the document path — each row becomes a text document that flows through
normal cognify — by declaring a document-source tag (see
dlt_utils.document_source_tag)."""

import json
import os
import shutil
import tempfile
from typing import Any, Callable, List, Optional, Set
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import UUID

from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

from .create_dlt_source import (
    is_connection_string,
    is_csv_path,
    is_csv_upload,
    create_dlt_source_from_connection_string,
)
from .config import get_ingestion_config
from .data_item import DataItem
from .dlt_row_data import DltRowData
from .dlt_utils import document_source_tag
from .ingest_dlt_source import ingest_dlt_source

logger = get_logger("resolve_dlt_sources")

# Cells longer than this never become ColumnValue nodes — long values are
# free text, not categorical data, and would produce useless one-off nodes.


async def resolve_dlt_sources(
    data: Any,
    dataset_name: str,
    user: User,
    dataset_id: UUID = None,
    **kwargs,
) -> Any:
    """Resolve DLT resources (and auto-detected structured data) into DataItems.

    Non-DLT items pass through unchanged. Each relational DLT resource is
    ingested via ``ingest_dlt_source`` and wrapped in a *single* manifest
    ``DataItem`` with a stable ``data_id``; the manifest carries all unique
    rows, their enriched text, and resolved FK references. Document-tagged
    sources instead yield one text-document DataItem per row, routed through
    normal cognify.

    Returns a ``(data, orphan_cleanup)`` tuple. ``data`` is the (possibly
    expanded) data — a single item, a list, or unchanged if nothing was DLT.
    ``orphan_cleanup`` is an async callable that deletes dlt rows no longer in
    the source, or ``None`` when there is nothing to clean up; the caller must
    await it *after* the fresh rows are committed.
    """
    # Lazy-import DLT types so the dlt package is not a hard dependency
    try:
        from dlt.extract import DltResource, SourceFactory
        from dlt.extract.source import DltSource
    except ImportError:
        # dlt not installed — nothing to resolve. Inputs that would have matched the
        # auto-detection below otherwise degrade to plain document ingestion.
        #
        # For a connection string that degradation is a credential leak, not just a
        # loss of function: the DSN is returned unchanged, ingested as a text
        # document, and written verbatim to disk by LocalFileStorage.store(). A DSN
        # like postgresql://user:pw@host/db -- the form .env.template itself uses --
        # lands in clear text on the filesystem, and the user does not get the table
        # contents they asked for either. Refuse instead.
        items = data if isinstance(data, list) else [data]
        _log_structured_inputs_without_dlt(items)
        if any(isinstance(item, str) and is_connection_string(item) for item in items):
            from cognee.exceptions import CogneeValidationError

            raise CogneeValidationError(
                message="A database connection string was passed but the 'dlt' extra is "
                "not installed, so it cannot be read. Install it with "
                "`pip install cognee[dlt]`. Refusing to ingest the connection string "
                "as a plain document: it would be stored on disk in clear text.",
                name="DltExtraNotInstalled",
            )
        return data, None

    primary_key = kwargs["primary_key"] if "primary_key" in kwargs else None
    write_disposition = kwargs["write_disposition"] if "write_disposition" in kwargs else "replace"
    query = kwargs["query"] if "query" in kwargs else None
    max_rows_per_table = kwargs.get("max_rows_per_table")
    column_value_columns = kwargs.get("column_value_columns")

    # Normalise to list for uniform processing
    data_list = data if isinstance(data, list) else [data]

    # --- Auto-detect structured data (connection strings) ------------------
    # CSVs are NOT handled here anymore: they route through the loader engine
    # (dlt_csv_loader, registered above csv_loader) inside the ingest
    # pipeline, which owns localization (s3/file://) and manifest emission.
    data_list = _normalize_structured_inputs(data_list, query)

    dlt_items = []
    non_dlt_items = []

    for item in data_list:
        if isinstance(item, (DltResource, DltSource, SourceFactory)):
            dlt_items.append(item)
        else:
            non_dlt_items.append(item)

    if not dlt_items:
        # Nothing to expand — return original data unchanged
        return data, None

    # A dlt source may opt into the document path (each row → a text document
    # that goes through normal cognify) by declaring a document-source tag;
    # every other dlt source takes the relational manifest path below.
    document_items = [i for i in dlt_items if document_source_tag(i)]
    relational_items = [i for i in dlt_items if not document_source_tag(i)]

    # --- Document-tagged sources: one text document per row -----------------
    # Document sources honour the caller's write_disposition so both sync models
    # work: "replace" for delete-feed-less snapshot sources (Notion/Slack — each
    # run rewrites staging with exactly the rows currently visible) and "merge"
    # (+ a hard_delete tombstone column) for incremental sources with a real
    # delete feed (Google Drive's Changes API). Either way the whole set is read
    # back (max_rows_per_table=0) so orphan cleanup can forget rows that dropped
    # out of the current corpus. write_disposition/primary_key default to
    # "replace"/"id" (see the kwargs resolution above).
    document_data_items: list[DataItem] = []
    document_fresh_ids: Set[UUID] = set()
    document_source_tags: set[str] = set()
    for dlt_item in document_items:
        source_tag = document_source_tag(dlt_item)
        document_source_tags.add(source_tag)
        rows = await ingest_dlt_source(
            dlt_item,
            dataset_name,
            primary_key=primary_key or "id",
            write_disposition=write_disposition,
            max_rows_per_table=0,
        )
        # Dataset-scoped ids with the pre-scoping adoption probe: rows are
        # dataset-scoped with id as primary key, so a dataset-free
        # derivation would pin the same id when one source loads into two
        # datasets — a hard collision on the foreign-pin guard.
        row_ids = await _stable_row_ids(rows, user, dataset_id)
        for row, data_id in zip(rows, row_ids):
            document_fresh_ids.add(data_id)
            document_data_items.append(_build_document_data_item(row, data_id, source_tag))

    # --- Relational sources: one manifest DataItem per source -----------
    expanded_items: list[DataItem] = []
    manifest_data_ids: Set[UUID] = set()
    for dlt_item in relational_items:
        rows = await ingest_dlt_source(
            dlt_item,
            dataset_name,
            primary_key=primary_key,
            write_disposition=write_disposition,
            max_rows_per_table=max_rows_per_table,
        )
        source_name = getattr(dlt_item, "name", None) or dataset_name
        item = await _build_source_manifest_item(
            rows,
            source_name,
            dataset_name,
            user,
            column_value_columns=column_value_columns,
        )
        if item is not None:
            if item.data_id in manifest_data_ids:
                # Stable ids are seeded from (dataset, source name): two
                # resources sharing a name would silently overwrite each
                # other's manifest. Fail loudly instead of last-write-wins.
                raise ValueError(
                    f"Two DLT sources in one add() resolve to the same identity "
                    f"(dataset {dataset_name!r}, source {source_name!r}). "
                    "Give each source a unique name."
                )
            expanded_items.append(item)
            manifest_data_ids.add(item.data_id)

    # Manifest source names, collected before document items join the list.
    ingested_source_names = {item.system_metadata["source_name"] for item in expanded_items}
    expanded_items.extend(document_data_items)

    logger.info(
        "Resolved %d DLT source(s) into %d DataItem(s) (%d manifest(s), %d document(s)).",
        len(dlt_items),
        len(expanded_items),
        len(manifest_data_ids),
        len(document_data_items),
    )

    # --- Prepare deferred orphan cleanup ------------------------------------
    # Deletion of orphaned dlt rows is deferred to *after* the fresh rows are
    # committed by the add pipeline, to avoid a data-loss window: if ingestion
    # failed between deletion and commit, the orphans would be gone and the
    # replacements never stored. We return a cleanup coroutine for the caller
    # to await post-commit instead of deleting here.
    #
    # Relational sources skip cleanup for "append" (each run intentionally adds
    # new rows). Document sources reconcile the full current snapshot, so rows
    # missing from it (deleted/unshared) must be forgotten. The two paths are
    # cleaned separately so an append relational run never treats document rows
    # as orphans, or vice versa.
    #
    # Both paths skip cleanup when their fresh set is empty: an empty read-back
    # cannot be distinguished from a failed/misconfigured sync, and treating it
    # as "everything is an orphan" would wipe the whole corpus. Leaving stale
    # rows for one cycle is the safe failure mode.
    do_manifest_cleanup = write_disposition != "append" and bool(manifest_data_ids)
    do_document_cleanup = bool(document_fresh_ids)

    orphan_cleanup: Optional[Callable[[], Any]] = None
    if do_manifest_cleanup or do_document_cleanup:

        async def _cleanup() -> None:
            if do_manifest_cleanup:
                await _delete_dlt_orphans(
                    dataset_name,
                    user,
                    manifest_data_ids,
                    manifest_source_names=ingested_source_names,
                )
            if do_document_cleanup:
                await _delete_dlt_orphans(
                    dataset_name, user, document_fresh_ids, sources=tuple(document_source_tags)
                )

        orphan_cleanup = _cleanup

    result = non_dlt_items + expanded_items
    return result, orphan_cleanup


def _normalize_structured_inputs(data_list: list, query: Optional[str]) -> list:
    """Wrap auto-detected structured inputs in dlt sources.

    Connection strings become dlt sql_database sources; everything else
    passes through unchanged. CSV inputs are deliberately NOT handled here —
    the loader engine's dlt_csv_loader owns them inside the ingest pipeline
    (with localization of s3:// and file:// locations built in).
    """
    normalized = []
    for item in data_list:
        if isinstance(item, str) and is_connection_string(item):
            normalized.append(create_dlt_source_from_connection_string(item, query=query))
        else:
            normalized.append(item)
    return normalized


def _log_structured_inputs_without_dlt(data_list: list) -> None:
    """Warn when inputs that would auto-route to the DLT path are about to be
    ingested as plain documents because dlt is not installed. Only counts are
    logged — the inputs themselves may embed credentials (connection strings)."""
    csv_paths = 0
    connection_strings = 0
    csv_uploads = 0
    for item in data_list:
        if isinstance(item, str) and is_csv_path(item):
            csv_paths += 1
        elif isinstance(item, str) and is_connection_string(item):
            connection_strings += 1
        elif is_csv_upload(item):
            csv_uploads += 1
    total = csv_paths + connection_strings + csv_uploads
    if total:
        logger.warning(
            "dlt is not installed: %d structured input(s) (%d CSV path(s), %d connection "
            "string(s), %d CSV upload(s)) will be ingested as plain documents through the "
            "standard LLM pipeline instead of the DLT manifest path. Install the dlt extra "
            '(pip install "cognee[dlt]") to route them through DLT.',
            total,
            csv_paths,
            connection_strings,
            csv_uploads,
        )


async def _build_source_manifest_item(
    rows: List[DltRowData],
    source_name: str,
    dataset_name: str,
    user: User,
    column_value_columns: Optional[dict] = None,
) -> Optional[DataItem]:
    """Build a single manifest DataItem describing a whole DLT source.

    Rows are deduplicated by identity (table, pk_value, content_hash) — DLT
    child tables can contain rows that are byte-identical once dlt bookkeeping
    columns are stripped. Each unique row keeps a stable ``node_id`` (the same
    id the legacy per-row ingestion used as its Data id) so the graph/vector
    node ids match the previous per-row pipeline output.

    ``column_value_columns`` selects cells to record for ColumnValue node
    emission ({"table": ["col", ...]}, "*" wildcards); None falls back to the
    ``dlt_column_value_columns`` ingestion config setting.
    """
    if not rows:
        return None

    if column_value_columns is None:
        column_value_columns = get_ingestion_config().dlt_column_value_columns

    dlt_db_name = rows[0].dlt_db_name
    unique_rows = _dedupe_rows(rows, source_name)

    # Stable per-row node ids via the shared identifier formula. FK lookup
    # maps (table, pk_value) → node_id; when multiple rows share a PK value,
    # the last one wins (best-effort).
    # TODO: add a batch variant of get_unique_data_id (one query per source
    # instead of one per row) after the incremental load PR reshapes it.
    node_ids: dict[tuple[str, str, str], UUID] = {
        key: await get_unique_data_id(_dlt_row_identifier(row), user)
        for key, row in unique_rows.items()
    }

    fk_lookup: dict[tuple[str, str], UUID] = {}
    for key, row in unique_rows.items():
        node_id = node_ids[key]
        fk_key = (row.table_name, row.primary_key_value)
        existing = fk_lookup.get(fk_key)
        if existing is not None and existing != node_id:
            logger.warning(
                "Duplicate primary key during FK resolution: table=%s pk=%s. "
                "FK edges targeting this key resolve to the last row "
                "(content_hash=%s); earlier rows with this key are shadowed.",
                row.table_name,
                row.primary_key_value,
                row.content_hash,
            )
        fk_lookup[fk_key] = node_id

    tables: dict[str, dict] = {}
    manifest_rows: list[dict] = []
    # (source_table, fk_column, ref_table, fk_value) for FKs whose target row
    # was not loaded — collected here and reported once after the loop.
    missing_fk_targets: list[tuple[str, str, str, str]] = []
    for key, row in unique_rows.items():
        if row.table_name not in tables:
            tables[row.table_name] = {
                "schema_info": row.schema_info,
                "foreign_keys": row.foreign_keys,
                "dlt_db_name": row.dlt_db_name,
            }

        manifest_row = {
            "node_id": str(node_ids[key]),
            "table_name": row.table_name,
            "primary_key_column": row.primary_key_column,
            "primary_key_value": row.primary_key_value,
            "content_hash": row.content_hash,
            "text": _build_schema_context_text(row),
            "fk_references": _resolve_fk_references(row, fk_lookup, missing_fk_targets),
        }
        column_values = _selected_column_values(row, column_value_columns)
        if column_values:
            manifest_row["column_values"] = column_values
        manifest_rows.append(manifest_row)

    if missing_fk_targets:
        sample = ", ".join(
            f"{src}.{col} -> {ref}:{val}" for src, col, ref, val in missing_fk_targets[:5]
        )
        logger.warning(
            "%d foreign key reference(s) could not be resolved to a loaded row "
            "and were dropped (no edge created). The target row was likely not "
            "ingested (e.g. it is beyond max_rows_per_table). "
            "Sample (source_table.column -> ref_table:value): %s",
            len(missing_fk_targets),
            sample,
        )

    # Deterministic ordering so the manifest content hash is stable across
    # re-ingests of unchanged data (content change drives reprocessing).
    manifest_rows.sort(key=lambda r: (r["table_name"], r["node_id"]))

    manifest = {
        "version": 1,
        "source_name": source_name,
        "tables": tables,
        "rows": manifest_rows,
    }
    manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)

    # The data_id is STABLE — seeded from (dataset, source name) with no
    # content hash — so a changed source updates its Data row in place instead
    # of orphaning it (which left the source absent from the graph between add
    # and cognify). add() is idempotent: a re-add of the same source — changed
    # or not — keeps the completed record as-is. Updating an existing source
    # is update()'s job (explicit UUID), or an explicit re-ingest via
    # add(..., incremental_loading=False, data_cache=False). Renaming a source (or dataset)
    # changes this identity and is remove + add — a one-time full rebuild,
    # by design.
    data_id = await get_unique_data_id(f"dlt_source:{dataset_name}:{source_name}", user)

    return DataItem(
        data=manifest_text,
        label=f"dlt_source:{source_name}",
        system_metadata={
            "source": "dlt_source",
            "source_name": source_name,
            "dlt_db_name": dlt_db_name,
            "tables": sorted(tables),
            "row_count": len(manifest_rows),
        },
        data_id=data_id,
    )


# ---------------------------------------------------------------------------
# Helpers (moved from ingest_data.py)
# ---------------------------------------------------------------------------


def _dedupe_rows(rows: List[DltRowData], source_name: str) -> dict[tuple, DltRowData]:
    """Deduplicate rows by identity (table, pk_value, content_hash).

    DLT child tables can contain rows that are byte-identical once dlt
    bookkeeping columns are stripped. The last occurrence wins, matching the
    previous FK-target behavior for duplicate keys.
    """
    unique_rows: dict[tuple[str, str, str], DltRowData] = {}
    duplicates = 0
    for row in rows:
        key = (row.table_name, row.primary_key_value, row.content_hash)
        if key in unique_rows:
            duplicates += 1
        unique_rows[key] = row

    if duplicates:
        logger.info(
            "Source '%s': collapsed %d duplicate row(s) into %d unique row(s).",
            source_name,
            duplicates,
            len(unique_rows),
        )

    return unique_rows


def _selected_column_values(dlt_row: DltRowData, selection: Optional[dict]) -> dict:
    """Pick row cells that should become shared ColumnValue graph nodes.

    ``selection`` maps table name to a column list; "*" is a wildcard for
    either side. Primary-key and foreign-key columns are excluded (they are
    row identity and edges already), as are null and empty values. A positive
    ``dlt_max_column_value_length`` (default 0 = unlimited) additionally
    skips values longer than the bound.
    """
    if not selection:
        return {}
    columns = selection.get(dlt_row.table_name) or selection.get("*")
    if not columns:
        return {}
    take_all = "*" in columns
    max_value_length = get_ingestion_config().dlt_max_column_value_length

    fk_columns = {fk.get("column", "") for fk in dlt_row.foreign_keys}
    picked = {}
    for column, value in dlt_row.row_data.items():
        if column == dlt_row.primary_key_column or column in fk_columns:
            continue
        if not take_all and column not in columns:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if max_value_length and len(text) > max_value_length:
            continue
        picked[column] = text
    return picked


def _dlt_row_identifier(row: DltRowData) -> str:
    """Stable identifier for a dlt row, used to derive a deterministic data_id.

    Includes ``content_hash`` so that unchanged rows keep the same data_id
    across runs (no re-ingest/re-cognify) while edited rows get a fresh one.
    """
    return f"dlt:{row.table_name}:{row.primary_key_value}:{row.content_hash}"


async def _stable_row_ids(rows: List[DltRowData], user: User, dataset_id: UUID) -> List[UUID]:
    """Dataset-scoped stable ids for dlt rows, adopting pre-scoping rows in place.

    Ids are derived from (dataset, table, pk, content_hash, user), so the same
    source loaded into two datasets yields two independent id families and
    dataset-scoped rows never collide across datasets (a cross-dataset
    collision would trip ingestion's foreign-pin guard). Rows ingested before
    ids were dataset-namespaced carry the old derivation as their primary id;
    those are adopted — the old id kept — ONLY when the row lives in this
    dataset, so unchanged rows stay stable through the upgrade instead of
    being re-minted (and orphan-cleanup never mistakes them for stale rows).
    A row of another dataset never matches the probe.
    """
    if not rows:
        return []

    new_ids = [await get_unique_data_id(_dlt_row_identifier(row), user, dataset_id) for row in rows]
    old_ids = [await get_unique_data_id(_dlt_row_identifier(row), user) for row in rows]

    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        adopted = set(
            (
                await session.execute(
                    select(Data.id).filter(Data.id.in_(old_ids), Data.dataset_id == dataset_id)
                )
            ).scalars()
        )
    return [old_id if old_id in adopted else new_id for old_id, new_id in zip(old_ids, new_ids)]


def _build_document_data_item(row: DltRowData, data_id: UUID, source_tag: str) -> DataItem:
    """Build a text-document DataItem from a document-source dlt row.

    The row is expected to carry ``title``/``content`` columns (and optionally
    ``url``/``id``). Tagging ``system_metadata["source"] = source_tag``
    routes the document through normal cognify entity extraction rather
    than the deterministic manifest path.
    """
    row_data = row.row_data
    title = _clean(row_data.get("title"))
    content = _clean(row_data.get("content"))
    text = f"# {title}\n\n{content}".strip() if title else content

    system_metadata = {"source": source_tag, "title": title or None}
    if row_data.get("url"):
        system_metadata["url"] = row_data["url"]
    if row_data.get("id"):
        system_metadata["external_id"] = str(row_data["id"])

    return DataItem(
        data=text,
        label=title or str(row_data.get("id")),
        system_metadata=system_metadata,
        data_id=data_id,
    )


def _clean(value: Any) -> str:
    """Coerce a possibly-None cell value to a stripped string."""
    return str(value).strip() if value is not None else ""


def _build_schema_context_text(dlt_row: DltRowData) -> str:
    """Build a human-readable, schema-enriched text representation of a DLT row.

    This text is stored as the document content and used for vector search.
    DLT rows bypass LLM extraction — their graph is built deterministically
    from the relational schema by ``extract_dlt_source_edges``.
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


def _resolve_fk_references(
    dlt_row: DltRowData,
    row_id_lookup: dict,
    missing_targets: Optional[list] = None,
) -> list:
    """Resolve foreign key columns to target row node ids for graph edge creation.

    Returns a list of dicts:
    [{"column": "dept_id", "target_table": "departments", "target_pk_value": "10",
      "target_node_id": "uuid-string", "relationship_name": "dept_id_references_departments"}]

    When ``missing_targets`` is provided, FK references whose target row was not
    loaded are appended to it as ``(source_table, column, ref_table, value)`` so
    the caller can report the dropped edges instead of silently losing them.
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
        target_node_id = row_id_lookup.get(target_key)

        if target_node_id is not None:
            relationship_name = f"{fk_column}_references_{ref_table}"
            references.append(
                {
                    "column": fk_column,
                    "target_table": ref_table,
                    "target_column": ref_column,
                    "target_pk_value": fk_value_str,
                    "target_node_id": str(target_node_id),
                    "relationship_name": relationship_name,
                }
            )
        elif missing_targets is not None:
            missing_targets.append((dlt_row.table_name, fk_column, ref_table, fk_value_str))

    return references


async def _delete_dlt_orphans(
    dataset_name: str,
    user: User,
    fresh_data_ids: Set[UUID],
    # "dlt" stays in the sweep purely as residue cleanup: pre-manifest per-row
    # records are unsupported (classification raises on them), and re-adding a
    # source deletes any that linger.
    sources: tuple[str, ...] = ("dlt", "dlt_source"),
    manifest_source_names: Optional[Set[str]] = None,
) -> None:
    """Delete dlt-sourced Data records (and their graph/vector artifacts) that
    are no longer present in the freshly-ingested dlt sources.

    This handles the case where rows are deleted from the upstream database
    and the user re-ingests.  dlt cleans its own staging DB, but cognee's
    relational, graph, and vector stores still hold stale data.

    ``sources`` restricts cleanup to Data whose ``system_metadata["source"]``
    is one of the given tags: the default covers relational manifests and
    legacy per-row records, while document cleanup passes its document tags
    (e.g. ``("notion",)``) so reconciling one path never removes the other's
    records. For manifests, ``manifest_source_names`` additionally restricts
    candidates to those source names — a dataset can hold several DLT sources,
    and re-ingesting one must not delete the others. Legacy per-row records
    (source == "dlt") predate source attribution and are always migrated away.
    """
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.data.methods.delete_data import delete_data
    from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
        delete_data_nodes_and_edges,
    )
    from cognee.context_global_variables import set_database_global_context_variables

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
        # Deletion decisions key on system_metadata ONLY — external_metadata
        # is user-writable and must never make a record deletable.
        meta = data_item.system_metadata
        if not isinstance(meta, dict):
            continue
        source = meta.get("source")
        if source not in sources:
            continue
        if (
            source == "dlt_source"
            and manifest_source_names is not None
            and meta.get("source_name") not in manifest_source_names
        ):
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

    failed: list = []
    # The graph + vector engines are dataset-scoped under access control, so run
    # the per-orphan delete inside the dataset DB context (mirrors the canonical
    # datasets.delete_data routing). Without it, cleanup resolves the *default*
    # engines — the background-ingest path invokes orphan_cleanup before any
    # pipeline establishes the context — so the graph + vector purge silently
    # targets the wrong DB and leaves the forgotten row's chunks/entities
    # searchable under ENABLE_BACKEND_ACCESS_CONTROL.
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        for orphan in orphans:
            try:
                # Called unconditionally: gating on has_data_related_nodes would
                # skip graphs whose provenance is stamped in-graph instead of the
                # relational ledger (the function handles both paths and no-ops
                # when nothing is related).
                deleted_elements = await delete_data_nodes_and_edges(dataset.id, orphan.id, user.id)
                await delete_data(orphan, dataset.id)
                # Invalidate session answers that used the orphan's graph
                # elements (best-effort; must not fail the cleanup).
                try:
                    from cognee.modules.session_lifecycle.invalidate_sessions import (
                        invalidate_sessions_for_deleted_data,
                    )

                    await invalidate_sessions_for_deleted_data(
                        dataset.id,
                        deleted_elements.node_ids,
                        deleted_elements.edge_ids,
                        user_id=user.id,
                    )
                except Exception:
                    logger.warning(
                        "Session invalidation after orphan cleanup failed (non-fatal).",
                        exc_info=True,
                    )
            except Exception:
                failed.append(orphan.id)
                logger.warning(
                    "Failed to delete orphaned dlt row data_id=%s, skipping.",
                    orphan.id,
                    exc_info=True,
                )

    if failed:
        # Surface partial-cleanup failures loudly: the stale rows remain across
        # the relational, graph, and vector stores and will be retried on the
        # next ingest. We log rather than raise — the fresh rows are already
        # committed by this point and best-effort cleanup should not fail an
        # otherwise-successful add.
        logger.error(
            "Failed to delete %d of %d orphaned dlt row(s) from dataset '%s'. "
            "Stale data remains and will be retried on the next ingest. data_ids=%s",
            len(failed),
            len(orphans),
            dataset_name,
            ", ".join(str(i) for i in failed),
        )
