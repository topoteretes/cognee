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
from collections.abc import Callable
from typing import Any, NoReturn
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import UUID

from cognee.modules.data.methods.get_existing_data_ids import (
    DATA_ID_BATCH_SIZE,
    get_existing_data_ids,
)
from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id, get_unique_data_ids
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

from .config import get_ingestion_config
from .create_dlt_source import (
    create_dlt_source_from_connection_string,
    is_connection_string,
    is_csv_path,
    is_csv_upload,
)
from .data_item import DataItem
from .dlt_row_data import DltRowData
from .dlt_temporal import temporal_cells
from .dlt_utils import (
    NODE_SET_COLUMN,
    SOURCE_LABEL_ATTR,
    STRUCTURE_COLUMN,
    column_selected,
    document_source_tag,
)
from .ingest_dlt_source import ingest_dlt_source

logger = get_logger("resolve_dlt_sources")

# Cells longer than this never become ColumnValue nodes — long values are
# free text, not categorical data, and would produce useless one-off nodes.


def dlt_manifest_identifier(dataset_name: str, source_name: str) -> str:
    """The identity a DLT source's manifest is seeded from: (dataset, source name)."""
    return f"dlt_source:{dataset_name}:{source_name}"


def is_dlt_input(item: Any) -> bool:
    """Whether ``item`` is a dlt resource, source, or source factory."""
    try:
        from dlt.extract import DltResource, SourceFactory
        from dlt.extract.source import DltSource
    except ImportError:
        return False
    return isinstance(item, (DltResource, DltSource, SourceFactory))


async def check_dlt_replacement(
    dlt_item: Any, pinned_id: UUID, dataset_name: str, user: User
) -> None:
    """Refuse a dlt source that cannot replace the document ``pinned_id``.

    update()'s full rebuild drops the document's memory and re-adds the
    replacement pinned to its id, so this runs BEFORE that: a replacement that
    would not resolve to the same manifest must be refused while the
    document's memory still exists. A document-tagged source yields one document per
    row and has no single identity; a relational source under another name
    resolves to a different manifest, and rebuilding it under the old id
    would leave the next plain add() of that source minting a second one.
    """
    from cognee.exceptions import CogneeValidationError

    if document_source_tag(dlt_item):
        raise CogneeValidationError(
            message=(
                "A document-tagged DLT source yields one document per row, so it cannot "
                "replace a single document; update() cannot take it. Re-add the source."
            ),
            name="DltDocumentSourceNotUpdatable",
        )
    source_name = getattr(dlt_item, "name", None) or dataset_name
    manifest_id = await get_unique_data_id(dlt_manifest_identifier(dataset_name, source_name), user)
    if manifest_id != pinned_id:
        raise CogneeValidationError(
            message=(
                f"DLT source {source_name!r} resolves to manifest {manifest_id}, not the "
                f"document {pinned_id} being updated. update() replaces a DLT source under "
                "the same source name."
            ),
            name="DltSourceIdentityMismatch",
        )


async def resolve_dlt_sources(
    data: Any,
    dataset_name: str,
    user: User,
    dataset_id: UUID | None = None,
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

    primary_key = kwargs.get("primary_key", None)
    write_disposition = kwargs.get("write_disposition", "replace")
    query = kwargs.get("query", None)
    max_rows_per_table = kwargs.get("max_rows_per_table")
    column_value_columns = kwargs.get("column_value_columns")
    temporal_columns = kwargs.get("temporal_columns")

    # Normalise to list for uniform processing
    data_list = data if isinstance(data, list) else [data]

    # --- Auto-detect structured data (connection strings) ------------------
    # CSVs are NOT handled here anymore: they route through the loader engine
    # (dlt_csv_loader, registered above csv_loader) inside the ingest
    # pipeline, which owns localization (s3/file://) and manifest emission.
    data_list = _normalize_structured_inputs(data_list, query)

    dlt_items = []
    non_dlt_items = []
    # update()'s full rebuild re-adds the replacement wrapped in a DataItem
    # pinned to the manifest's id. Unwrap it here so the source is ingested
    # like a bare one, and remember the pin to check it against the manifest
    # identity the source resolves to.
    pinned_ids: dict[int, UUID] = {}

    for item in data_list:
        if isinstance(item, DataItem) and isinstance(
            item.data, (DltResource, DltSource, SourceFactory)
        ):
            if item.data_id is not None:
                pinned_ids[id(item.data)] = item.data_id
            item = item.data
        if isinstance(item, (DltResource, DltSource, SourceFactory)):
            dlt_items.append(item)
        else:
            non_dlt_items.append(item)

    if not dlt_items:
        # Nothing to expand — return original data unchanged
        return data, None

    # A pinned item must resolve to the manifest it is pinned to; update()
    # checks this before deleting the document, and it is re-checked here so
    # a pinned wrapper from any caller is held to the same rule.
    for dlt_item in dlt_items:
        if id(dlt_item) in pinned_ids:
            await check_dlt_replacement(dlt_item, pinned_ids[id(dlt_item)], dataset_name, user)

    # A dlt source may opt into the document path (each row → a text document
    # that goes through normal cognify) by declaring a document-source tag;
    # every other dlt source takes the relational manifest path below.
    document_items = [i for i in dlt_items if document_source_tag(i)]
    relational_items = [i for i in dlt_items if not document_source_tag(i)]

    # --- Document-tagged sources: one text document per row -----------------
    # Document sources honour the caller's write_disposition so both sync models
    # work: "replace" for snapshot sources (Slack, each run rewrites staging with
    # exactly the rows currently visible) and "merge" (+ a hard_delete tombstone
    # column) for incremental sources that emit their own deletions (Google
    # Drive's Changes API, Notion's re-walk against the ids it saw last run).
    # Either way the whole set is read back (max_rows_per_table=0) so orphan
    # cleanup can forget rows that dropped out of the current corpus.
    # write_disposition/primary_key default to "replace"/"id" (see the kwargs
    # resolution above).
    document_data_items: list[DataItem] = []
    document_fresh_ids: set[UUID] = set()
    document_scopes: set[tuple[str, str]] = set()
    for dlt_item in document_items:
        source_tag = document_source_tag(dlt_item)
        rows = await ingest_dlt_source(
            dlt_item,
            dataset_name,
            primary_key=primary_key or "id",
            write_disposition=write_disposition,
            max_rows_per_table=0,
        )
        loaded_tables = getattr(rows, "loaded_tables", {row.table_name for row in rows})
        document_scopes.update((source_tag, table) for table in loaded_tables)
        # Dataset-scoped ids with the pre-scoping adoption probe: rows are
        # dataset-scoped with id as primary key, so a dataset-free
        # derivation would pin the same id when one source loads into two
        # datasets — a hard collision on the foreign-pin guard.
        row_ids = await _stable_row_ids(rows, user, dataset_id)
        source_items = []
        for row, data_id in zip(rows, row_ids):
            document_fresh_ids.add(data_id)
            source_items.append(_build_document_data_item(row, data_id, source_tag))
        await _refresh_row_structure(source_items, dataset_id)
        document_data_items.extend(source_items)

    # --- Relational sources: one manifest DataItem per source -----------
    expanded_items: list[DataItem] = []
    manifest_data_ids: set[UUID] = set()
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
            temporal_columns=temporal_columns,
            source_label=getattr(dlt_item, SOURCE_LABEL_ATTR, None),
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
    # Document cleanup is scoped to successfully loaded/read tables. A table
    # emptied by tombstones must forget its final document; an incremental run
    # with no load jobs has no such evidence and must not delete anything.
    # Relational manifests retain their conservative empty-read behavior.
    do_manifest_cleanup = write_disposition != "append" and bool(manifest_data_ids)
    do_document_cleanup = bool(document_scopes)

    orphan_cleanup: Callable[[], Any] | None = None
    if do_manifest_cleanup or do_document_cleanup:

        async def _cleanup() -> None:
            if do_manifest_cleanup:
                await _delete_dlt_orphans(
                    dataset_name,
                    user,
                    manifest_data_ids,
                    manifest_source_names=ingested_source_names,
                    dataset_id=dataset_id,
                )
            if do_document_cleanup:
                await _delete_dlt_orphans(
                    dataset_name,
                    user,
                    document_fresh_ids,
                    sources=tuple({source for source, _ in document_scopes}),
                    document_scopes=document_scopes,
                    dataset_id=dataset_id,
                )

        orphan_cleanup = _cleanup

    result = non_dlt_items + expanded_items
    return result, orphan_cleanup


def _normalize_structured_inputs(data_list: list, query: str | None) -> list:
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
    rows: list[DltRowData],
    source_name: str,
    dataset_name: str,
    user: User,
    column_value_columns: dict | None = None,
    temporal_columns: dict | None = None,
    source_label: str | None = None,
) -> DataItem | None:
    """Build a single manifest DataItem describing a whole DLT source.

    ``source_label`` names the database a connection-string source reads
    (``SOURCE_LABEL_ATTR``); the manifest records it so the source's table nodes
    can name it.

    Rows are deduplicated by identity (table, pk_value, content_hash) — DLT
    child tables can contain rows that are byte-identical once dlt bookkeeping
    columns are stripped. Each unique row keeps a stable ``node_id`` (the same
    id the legacy per-row ingestion used as its Data id) so the graph/vector
    node ids match the previous per-row pipeline output.

    ``column_value_columns`` selects cells to record for ColumnValue node
    emission ({"table": ["col", ...]}, "*" wildcards); None falls back to the
    ``dlt_column_value_columns`` ingestion config setting. ``temporal_columns``
    selects the date cells that become Timestamp edges the same way (default
    setting ``dlt_temporal_columns``: every date cell).
    """
    if not rows:
        return None

    if column_value_columns is None:
        column_value_columns = get_ingestion_config().dlt_column_value_columns
    if temporal_columns is None:
        temporal_columns = get_ingestion_config().dlt_temporal_columns

    dlt_db_name = rows[0].dlt_db_name
    unique_rows = _dedupe_rows(rows, source_name)

    # Stable per-row node ids via the shared identifier formula. FK lookup
    # maps (table, pk_value) → node_id; when multiple rows share a PK value,
    # the last one wins (best-effort).
    node_ids: dict[tuple[str, str, str], UUID] = dict(
        zip(
            unique_rows,
            await get_unique_data_ids(
                [_dlt_row_identifier(row) for row in unique_rows.values()], user
            ),
        )
    )

    fk_lookup: dict[tuple[str, str], UUID] = {}
    shadowed: dict[str, int] = {}
    for key, row in unique_rows.items():
        fk_key = (row.table_name, row.primary_key_value)
        if fk_key in fk_lookup:
            shadowed[row.table_name] = shadowed.get(row.table_name, 0) + 1
        fk_lookup[fk_key] = node_ids[key]
    for table_name, count in shadowed.items():
        logger.warning(
            "Table '%s': %d rows share a primary key value with an earlier row; a foreign "
            "key to such a value resolves to the last of those rows.",
            table_name,
            count,
        )
    # The columns each table's rows are keyed on: an FK to those finds its target
    # row through fk_lookup; an FK to other columns through an index of their values.
    key_columns = {row.table_name: row.primary_key_columns for row in unique_rows.values()}
    rows_of: dict[str, list[tuple[DltRowData, UUID]]] = {}
    for key, row in unique_rows.items():
        rows_of.setdefault(row.table_name, []).append((row, node_ids[key]))
    column_indexes: dict[tuple[str, tuple[str, ...]], dict] = {}

    def column_lookup(table_name: str, columns: tuple[str, ...]) -> dict:
        if (table_name, columns) not in column_indexes:
            column_indexes[(table_name, columns)] = _column_index(
                rows_of.get(table_name, []), columns
            )
        return column_indexes[(table_name, columns)]

    tables: dict[str, dict] = {}
    manifest_rows: list[dict] = []
    # (source_table, fk_column, ref_table, fk_value) for FKs whose target row
    # was not loaded — collected here and reported once after the loop.
    missing_fk_targets: list[tuple[str, str, str, str]] = []
    # (source_table, fk_columns, ref_table, ref_columns) for FKs to a UNIQUE column
    # whose value several loaded rows hold.
    ambiguous_fks: set[tuple[str, str, str, str]] = set()
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
            "primary_key_column": ", ".join(row.primary_key_columns),
            "primary_key_value": row.primary_key_value,
            "content_hash": row.content_hash,
            "text": _build_schema_context_text(row),
            "fk_references": _resolve_fk_references(
                row, fk_lookup, missing_fk_targets, key_columns, column_lookup, ambiguous_fks
            ),
        }
        timestamps = temporal_cells(row, temporal_columns)
        if timestamps:
            manifest_row["timestamps"] = timestamps
        # A date cell that anchors to a Timestamp is already a shared node for
        # its value, so it is not a ColumnValue as well: one node per cell.
        column_values = {
            column: value
            for column, value in _selected_column_values(row, column_value_columns).items()
            if column not in timestamps
        }
        if column_values:
            manifest_row["column_values"] = column_values
        manifest_rows.append(manifest_row)

    _report_missing_fk_targets(missing_fk_targets, set(key_columns))
    for src, col, ref, ref_col in sorted(ambiguous_fks):
        logger.warning(
            "Foreign key %s.%s -> %s.%s skipped values that several rows of '%s' hold; "
            "a foreign key must reference a unique column, so none of them is linked.",
            src,
            col,
            ref,
            ref_col,
            ref,
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
    if source_label:
        manifest["source_label"] = source_label
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
    data_id = await get_unique_data_id(dlt_manifest_identifier(dataset_name, source_name), user)

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


def _dedupe_rows(rows: list[DltRowData], source_name: str) -> dict[tuple, DltRowData]:
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


def _selected_column_values(dlt_row: DltRowData, selection: dict | None) -> dict:
    """Pick row cells that should become shared ColumnValue graph nodes.

    ``selection`` maps table name to a column list; "*" is a wildcard for
    either side. Primary-key and foreign-key columns are excluded (they are
    row identity and edges already), as are null and empty values. A positive
    ``dlt_max_column_value_length`` (default 0 = unlimited) additionally
    skips values longer than the bound.
    """
    if not selection:
        return {}
    max_value_length = get_ingestion_config().dlt_max_column_value_length

    fk_columns = {fk.get("column", "") for fk in dlt_row.foreign_keys}
    picked = {}
    for column, value in dlt_row.row_data.items():
        if column in dlt_row.primary_key_columns or column in fk_columns:
            continue
        if not column_selected(selection, dlt_row.table_name, column):
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


async def _stable_row_ids(rows: list[DltRowData], user: User, dataset_id: UUID) -> list[UUID]:
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

    from cognee.infrastructure.databases.relational import get_relational_engine

    identifiers = [_dlt_row_identifier(row) for row in rows]
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        new_ids = await get_unique_data_ids(identifiers, user, dataset_id, session)
        old_ids = await get_unique_data_ids(identifiers, user, session=session)
        adopted = await get_existing_data_ids(session, old_ids, dataset_id)
    return [old_id if old_id in adopted else new_id for old_id, new_id in zip(old_ids, new_ids)]


async def _refresh_row_structure(items: list[DataItem], dataset_id: UUID) -> None:
    """Write the tree position onto rows that are already ingested. add() skips a
    record whose content is unchanged, so a moved row would otherwise keep its old
    parent. Only the ``structure`` key changes, and only where it differs."""
    wanted = {
        item.data_id: item.system_metadata["structure"]
        for item in items
        if "structure" in item.system_metadata
    }
    if not wanted:
        return

    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    ids = list(wanted)
    async with get_relational_engine().get_async_session() as session:
        for start in range(0, len(ids), DATA_ID_BATCH_SIZE):
            rows = await session.execute(
                select(Data).filter(
                    Data.id.in_(ids[start : start + DATA_ID_BATCH_SIZE]),
                    Data.dataset_id == dataset_id,
                )
            )
            for data in rows.scalars():
                if data.system_metadata.get("structure") != wanted[data.id]:
                    data.system_metadata = {**data.system_metadata, "structure": wanted[data.id]}
            # One transaction per batch: the first sync after the upgrade touches every row.
            await session.commit()
            session.expunge_all()


def _build_document_data_item(row: DltRowData, data_id: UUID, source_tag: str) -> DataItem:
    """Build a text-document DataItem from a document-source dlt row.

    The row is expected to carry ``title``/``content`` columns (and optionally
    ``url``/``id``). Tagging ``system_metadata["source"] = source_tag``
    routes the document through normal cognify entity extraction rather
    than the deterministic manifest path. A ``cognee_node_set`` column names
    the row's own node sets (``_row_node_set``); they ride on the DataItem and
    ingest_data unions them with the call-level node_set. A ``cognee_structure``
    column says where the row sits in its source's tree (``_row_structure``);
    it is kept in ``system_metadata["structure"]`` for the structure pass.

    ``literal_text=True`` because ``content`` is provider data cognee did not
    write: an untitled row (no "# title" prefix) whose content happens to be
    just a URL or an existing local path must still be stored as that text,
    not fetched or read as if the caller had passed it directly to ``add()``.
    """
    row_data = row.row_data
    title = _clean(row_data.get("title"))
    content = _clean(row_data.get("content"))
    text = f"# {title}\n\n{content}".strip() if title else content

    system_metadata = {
        "source": source_tag,
        "title": title or None,
        "table_name": row.table_name,
    }
    if row_data.get("url"):
        system_metadata["url"] = row_data["url"]
    if row_data.get("id"):
        system_metadata["external_id"] = str(row_data["id"])
    structure = _row_structure(row, source_tag)
    if structure is not None:
        if "external_id" not in system_metadata:
            # Without an external_id the structure pass could not find this row as a parent.
            _raise_structure_shape(row, source_tag, "a structure on a row with no id")
        system_metadata["structure"] = structure

    return DataItem(
        data=text,
        label=title or str(row_data.get("id")),
        system_metadata=system_metadata,
        data_id=data_id,
        literal_text=True,
        node_set=_row_node_set(row, source_tag),
    )


def _reserved_json_column(row: DltRowData, source_tag: str, column: str, raise_shape) -> Any:
    """The parsed value of a json-hinted reserved column, ``None`` when unset.

    The column is loaded with a json type hint, so dlt moves a value that is
    not a list or object (a bare string, a number) into a variant column such
    as ``<column>__v_text`` and leaves the column itself NULL. Reading only the
    column would then ingest the row without it and no sign of it, so a
    variant raises through ``raise_shape``. The value comes back as parsed json
    or, on SQLite, as its JSON text, which text that is not JSON also raises
    for.
    """
    variants = sorted(
        name
        for name, value in row.row_data.items()
        if name.startswith(f"{column}__v_") and value is not None
    )
    if variants:
        raise_shape(row, source_tag, f"a value dlt stored in {', '.join(variants)}")
    raw = row.row_data.get(column)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            # ValueError only, no RecursionError guard: the column is loaded with a
            # json type hint (DOCUMENT_COLUMN_HINTS) and dlt refuses a value nested
            # deep enough to break json.loads, failing at extract before the row is
            # stored (checked on dlt 1.28.2: 200 levels load, 1000 fail). On Postgres
            # the value comes back already parsed. There is no input to guard against.
            raise_shape(row, source_tag, "text that is not JSON")
    return raw


def _row_node_set(row: DltRowData, source_tag: str) -> list[str]:
    """The node sets a document row names in its ``cognee_node_set`` column.

    A row states its full current node sets, so a row without the column, or
    with an empty list, has none: ``[]``, never ``None``. Re-ingesting an
    existing document treats ``None`` as "keep the stored node sets" (the
    PATCH rule of update()), which would leave a page that lost all its labels
    in the node sets it was removed from; ``[]`` clears them.

    The column must hold a JSON list of names. It comes back as that list (a
    json column) or as its JSON text (SQLite). Any other shape raises: the
    column is loaded with a json type hint, so dlt moves a value that is not a
    list or object (a bare string, a number) into a variant column such as
    ``cognee_node_set__v_text`` and leaves this one NULL. Reading only this
    column would then ingest the row with none of its node sets and no sign
    of it, so the sync stops instead and names the source, table and row.

    Every name is namespaced under ``source_tag`` unless it already is, so
    provider data can never name one of cognee's own node sets (``skills``,
    ``user_context``, ...), and deduplicated by node-set identity, so two
    spellings of one node set (``Project A``, ``project_a``) keep the first.
    An entry that is not a non-empty string raises too, matching what
    ingest_data does for a node_set passed to add().
    """
    raw = _reserved_json_column(row, source_tag, NODE_SET_COLUMN, _raise_node_set_shape)
    if raw is None:
        return []
    if not isinstance(raw, list):
        _raise_node_set_shape(row, source_tag, f"a JSON {type(raw).__name__}")
    # No cap on the number or length of names, on purpose. A cap would cut a
    # row's group memberships silently: the document lands with only some of
    # its node sets and nobody is told. If a source ever floods the graph with
    # node sets, fail the sync loudly, naming the source, table and count.
    # Never truncate.
    if not all(isinstance(value, str) and value.strip() for value in raw):
        _raise_node_set_shape(row, source_tag, "an entry that is not a non-empty string")
    prefix = f"{source_tag}:"
    # One entry per node set: spellings that normalize to the same id keep the first.
    unique: dict = {}
    for value in raw:
        name = value.strip()
        name = name if name.startswith(prefix) else prefix + name
        try:
            key = NodeSet.id_for(name)
        except UnicodeEncodeError:
            _raise_node_set_shape(row, source_tag, "an entry that cannot be encoded as a name")
        unique.setdefault(key, name)
    return list(unique.values())


_STRUCTURE_ENTRY_KEYS = {"kind", "id", "name", "document"}


def _row_structure(row: DltRowData, source_tag: str) -> dict | None:
    """The row's ``cognee_structure`` (``dlt_utils.STRUCTURE_COLUMN``), or ``None`` when
    unset, which is not the empty list of a row at the top of its tree. Any other
    shape stops the sync, naming the row, like a bad node set does."""
    raw = _reserved_json_column(row, source_tag, STRUCTURE_COLUMN, _raise_structure_shape)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("ancestors"), list):
        _raise_structure_shape(row, source_tag, "a value without an ancestors list")
    # No check for text a backend cannot store: dlt refuses a lone surrogate at
    # extract (checked on 1.28.2), and a graph write that refuses a value fails loudly.
    for position, entry in enumerate(raw["ancestors"]):
        if (
            not isinstance(entry, dict)
            or set(entry) - _STRUCTURE_ENTRY_KEYS
            or not all(isinstance(entry.get(key), str) and entry[key] for key in ("kind", "id"))
            or not isinstance(entry.get("name", ""), str)
            or not isinstance(entry.get("document", False), bool)
        ):
            _raise_structure_shape(row, source_tag, "an ancestor that is not a valid entry")
        if entry.get("document") and position != len(raw["ancestors"]) - 1:
            _raise_structure_shape(row, source_tag, "an ancestor after a document ancestor")
        if entry.get("document") and entry["id"] == str(row.row_data.get("id")):
            _raise_structure_shape(row, source_tag, "the row itself as its parent")
    return {"ancestors": raw["ancestors"]}


def _raise_structure_shape(row: DltRowData, source_tag: str, found: str) -> NoReturn:
    """Stop the sync: a row's ``cognee_structure`` is not a valid tree position."""
    from cognee.exceptions import CogneeValidationError

    raise CogneeValidationError(
        message=(
            f"Source '{source_tag}' table '{row.table_name}' row '{row.primary_key_value}': "
            f"{STRUCTURE_COLUMN} must be an object with an ancestors list of "
            '{"kind", "id", "name"?, "document"?} entries, but it holds '
            f"{found}. Fix the source and re-sync the affected rows. "
            "No document from this sync was stored."
        ),
        name="DltStructureColumnInvalid",
    )


def _raise_node_set_shape(row: DltRowData, source_tag: str, found: str) -> NoReturn:
    """Stop the sync: a row's ``cognee_node_set`` is not a JSON list of names."""
    from cognee.exceptions import CogneeValidationError

    raise CogneeValidationError(
        message=(
            f"Source '{source_tag}' table '{row.table_name}' row '{row.primary_key_value}': "
            f"{NODE_SET_COLUMN} must be a JSON list of names, but it holds {found}. "
            'Emit a list, even for one name (["name"]), and re-sync the affected rows. '
            "No document from this sync was stored."
        ),
        name="DltNodeSetColumnNotAList",
    )


def _clean(value: Any) -> str:
    """Coerce a possibly-None cell value to a stripped string."""
    return str(value).strip() if value is not None else ""


def _build_schema_context_text(dlt_row: DltRowData) -> str:
    """Build the text of a DLT row: its table, then one line per cell.

    The text is the row's document content and what vector search embeds, so it
    holds only what is specific to the row; the table's columns and foreign keys
    are on its SchemaTable node. DLT rows bypass LLM extraction — their graph is
    built deterministically from the relational schema by
    ``extract_dlt_source_edges``.
    """
    lines = [f"Table: {dlt_row.table_name}", "Row Data:"]
    lines.extend(f"  {key}: {value}" for key, value in dlt_row.row_data.items())
    return "\n".join(lines)


def _resolve_fk_references(
    dlt_row: DltRowData,
    row_id_lookup: dict,
    missing_targets: list | None = None,
    key_columns: dict[str, list[str]] | None = None,
    column_lookup=None,
    ambiguous_fks: set | None = None,
) -> list:
    """Resolve foreign key columns to target row node ids for graph edge creation.

    Returns a list of dicts:
    [{"column": "dept_id", "target_table": "departments", "target_column": "id",
      "target_pk_value": "10", "target_node_id": "uuid-string",
      "relationship_name": "dept_id_references_departments"}]
    ``target_pk_value`` is the referenced value, which is the target's key value
    unless the FK references a UNIQUE column.

    When ``missing_targets`` is provided, FK references whose target row was not
    loaded are appended to it as ``(source_table, column, ref_table, value)`` so
    the caller can report the dropped edges instead of silently losing them.

    ``row_id_lookup`` finds a row by the whole key its table is keyed on, given per
    table in ``key_columns``, so the row's per-column FKs are first grouped into the
    keys they form (``_fk_keys``): a composite FK finds its row by all its columns,
    in the target's key order. An FK that references other columns (a UNIQUE
    column) finds its row through ``column_lookup(ref_table, ref_columns)``, a
    ``_column_index`` of the target's rows. A value several rows hold creates no
    edge and is recorded in ``ambiguous_fks`` as
    ``(source_table, columns, ref_table, ref_columns)``, each joined by ", ".

    Both lookups hold only the rows of this manifest, so an FK finds its target
    only when both tables are loaded by the same ``add()``. A target loaded by an
    earlier or a later ``add()`` is never linked, and nothing revisits this row
    when it arrives. Loading the tables separately does not help within one dataset
    either: every connection-string source is named ``sql_database``, so a second
    ``add()`` of the same database maps to the same manifest id and is skipped as
    already completed.
    """
    references = []
    key_columns = key_columns or {}
    for ref_table, pairs, target_key_columns in _fk_keys(dlt_row.foreign_keys, key_columns):
        fk_columns = [column for column, _ in pairs]
        ref_columns = [ref_column for _, ref_column in pairs]
        fk_values = [dlt_row.row_data.get(column) for column in fk_columns]
        if any(value is None for value in fk_values):
            continue

        # Joined like the target's primary_key_value (see _key_value).
        fk_value_str = ":".join(str(value) for value in fk_values)
        fk_column = ", ".join(fk_columns)
        if target_key_columns is not None:
            target_node_id = row_id_lookup.get((ref_table, fk_value_str))
        else:
            index = column_lookup(ref_table, tuple(ref_columns)) if column_lookup else {}
            target_node_id = index.get(fk_value_str)
            if target_node_id is None and fk_value_str in index:
                if ambiguous_fks is not None:
                    ambiguous_fks.add(
                        (dlt_row.table_name, fk_column, ref_table, ", ".join(ref_columns))
                    )
                continue

        if target_node_id is not None:
            # The bare table name, so Postgres' "<dataset>.<table>" names the edge as SQLite does.
            relationship_name = f"{'_'.join(fk_columns)}_references_{ref_table.split('.')[-1]}"
            references.append(
                {
                    "column": fk_column,
                    "target_table": ref_table,
                    "target_column": ", ".join(target_key_columns or ref_columns),
                    "target_pk_value": fk_value_str,
                    "target_node_id": str(target_node_id),
                    "relationship_name": relationship_name,
                }
            )
        elif missing_targets is not None:
            missing_targets.append((dlt_row.table_name, fk_column, ref_table, fk_value_str))

    return references


def _report_missing_fk_targets(missing_targets: list, loaded_tables: set[str]) -> None:
    """Warn about FK references that found no target row, by why they missed.

    ``missing_targets`` holds ``(source_table, column, ref_table, value)`` tuples
    from ``_resolve_fk_references``; ``loaded_tables`` are the tables with rows in
    this manifest. A key whose target table has no rows here gets one warning of
    its own: edges are made only within one ``add()``, so loading that table in a
    separate one never links these rows (see ``_resolve_fk_references``). Misses
    against a table that was loaded keep the row-limit hint, where it can apply.
    """
    unloaded: dict[tuple[str, str, str], int] = {}
    unresolved = []
    for src, col, ref, val in missing_targets:
        if ref in loaded_tables:
            unresolved.append((src, col, ref, val))
        else:
            unloaded[(src, col, ref)] = unloaded.get((src, col, ref), 0) + 1

    for (src, col, ref), count in sorted(unloaded.items()):
        logger.warning(
            "Foreign key %s.%s -> %s: %d row(s) get no edge, because this add() loaded "
            "no rows of '%s'. Foreign keys link rows only within one add(): load '%s' "
            "in the same add() as '%s' to link them.",
            src,
            col,
            ref,
            count,
            ref,
            ref,
            src,
        )
    if unresolved:
        sample = ", ".join(f"{src}.{col} -> {ref}:{val}" for src, col, ref, val in unresolved[:5])
        logger.warning(
            "%d foreign key reference(s) could not be resolved to a loaded row "
            "and were dropped (no edge created). The target row was likely not "
            "ingested (e.g. it is beyond max_rows_per_table). "
            "Sample (source_table.column -> ref_table:value): %s",
            len(unresolved),
            sample,
        )


def _fk_keys(foreign_keys: list, key_columns: dict[str, list[str]]):
    """Group per-column FKs into the keys they form against their target's key.

    dlt lists every FK to one table as aligned column pairs, a composite FK's
    columns next to each other, so consecutive pairs to one table that together
    cover the target's key columns are one FK. Yields
    ``(ref_table, [(column, ref_column), ...], target_key_columns)`` with the pairs
    in the target's key order.

    ``target_key_columns`` is None for pairs that reference columns the target's
    rows are not keyed on (a UNIQUE column, or part of a composite key); the row
    is then found by those columns' values (``_column_index``). Consecutive such
    pairs to one table are one group, the way dlt lists a composite UNIQUE key.
    dlt's merged hints cannot tell that from separate single-column FKs that sit
    next to each other; matching them together only finds fewer rows, never a
    wrong one. A target with no loaded rows has no known key, and its pairs are
    taken one column each, to be reported as unresolved.
    """
    open_keys: dict[str, list[tuple[str, str]]] = {}
    open_columns: dict[str, list[tuple[str, str]]] = {}

    def close(groups, ref_table):
        pairs = groups.pop(ref_table, None)
        if pairs:
            yield ref_table, pairs, None

    for fk in foreign_keys:
        # Hints are written by _dlt_key_hints with all three fields; one without
        # them is a bug, not a key to skip.
        column, ref_table, ref_column = fk["column"], fk["ref_table"], fk["ref_column"]
        if not column or not ref_table or not ref_column:
            raise ValueError(f"Foreign key hint {fk!r} does not name its column, table and target")
        key = key_columns.get(ref_table)
        if key is None:
            yield ref_table, [(column, ref_column)], [ref_column]
            continue
        if ref_column not in key:
            yield from close(open_keys, ref_table)
            if any(ref_column == taken for _, taken in open_columns.get(ref_table, [])):
                yield from close(open_columns, ref_table)
            open_columns.setdefault(ref_table, []).append((column, ref_column))
            continue
        yield from close(open_columns, ref_table)
        pairs = open_keys.setdefault(ref_table, [])
        if any(ref_column == taken for _, taken in pairs):
            yield from close(open_keys, ref_table)
            pairs = open_keys.setdefault(ref_table, [])
        pairs.append((column, ref_column))
        if len(pairs) == len(key):
            open_keys.pop(ref_table)
            by_key_column = {taken: column for column, taken in pairs}
            yield ref_table, [(by_key_column[k], k) for k in key], key

    for groups in (open_keys, open_columns):
        for ref_table in list(groups):
            yield from close(groups, ref_table)


def _column_index(rows: list[tuple[DltRowData, UUID]], columns: tuple[str, ...]) -> dict:
    """Each value of ``columns`` among ``rows``, joined like a key value, to its row's
    node id, or to None when several rows hold it.

    A foreign key may only reference a primary key or a UNIQUE column set, so one
    loaded row holding the value is the referenced row. Several means the data
    breaks that constraint, and no row is the right one.
    """
    index: dict[str, UUID | None] = {}
    for row, node_id in rows:
        values = [row.row_data.get(column) for column in columns]
        if any(value is None for value in values):
            continue
        value = ":".join(str(v) for v in values)
        index[value] = None if value in index else node_id
    return index


async def _delete_dlt_orphans(
    dataset_name: str,
    user: User,
    fresh_data_ids: set[UUID],
    # "dlt" stays in the sweep purely as residue cleanup: pre-manifest per-row
    # records are unsupported (classification raises on them), and re-adding a
    # source deletes any that linger.
    sources: tuple[str, ...] = ("dlt", "dlt_source"),
    manifest_source_names: set[str] | None = None,
    document_scopes: set[tuple[str, str]] | None = None,
    dataset_id: UUID | None = None,
) -> None:
    """Delete dlt-sourced Data records (and their graph/vector artifacts) that
    are no longer present in the freshly-ingested dlt sources.

    This handles the case where rows are deleted from the upstream database
    and the user re-ingests.  dlt cleans its own staging DB, but cognee's
    relational, graph, and vector stores still hold stale data.

    ``dataset_id`` is the dataset the rows were just written to and is
    authoritative when given: a name resolves only among the caller's own
    datasets, so for a shared dataset written by id a name lookup would pick
    the caller's own same-named dataset (and sweep its rows) or nothing at all.

    ``sources`` restricts cleanup to Data whose ``system_metadata["source"]``
    is one of the given tags: the default covers relational manifests and
    legacy per-row records, while document cleanup passes its document tags
    (e.g. ``("notion",)``) so reconciling one path never removes the other's
    records. For manifests, ``manifest_source_names`` additionally restricts
    candidates to those source names — a dataset can hold several DLT sources,
    and re-ingesting one must not delete the others. Legacy per-row records
    (source == "dlt") predate source attribution and are always migrated away.

    Document scopes pair the provider tag with its staging table, so syncing
    one selected Drive folder cannot sweep another folder's documents. Legacy
    documents without a table stamp are retained: their owning scope cannot be
    proved, and deleting another source's data is worse than retaining residue.
    """
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.locks import dataset_lock
    from cognee.modules.conflicts.methods import retire_forgotten_decisions
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.data.methods.delete_data import delete_data
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
        delete_data_nodes_and_edges,
    )
    from cognee.modules.users.exceptions import PermissionDeniedError

    # Find the dataset — if it doesn't exist yet this is a first ingestion,
    # so there can be no orphans. Prefer the id: add() already resolved and
    # authorized it, and a name only matches datasets the caller owns.
    # Cleanup deletes, so it needs delete permission; a write-only grantee's
    # rows are committed by now, so skip the sweep instead of failing add().
    try:
        existing_datasets = await get_authorized_existing_datasets(
            user=user,
            permission_type="delete",
            datasets=[dataset_id] if dataset_id is not None else [dataset_name],
        )
    except PermissionDeniedError:
        logger.warning(
            "Skipping dlt orphan cleanup for dataset '%s': no delete permission, "
            "so rows deleted upstream stay in the dataset.",
            dataset_name,
        )
        return
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
        if document_scopes is not None and (source, meta.get("table_name")) not in document_scopes:
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
    # searchable under ENABLE_BACKEND_ACCESS_CONTROL. The dataset lock comes
    # first, as in datasets.delete_data: retiring conflict decisions rewrites
    # conflict nodes that a pipeline or a resolve_conflict call may be writing.
    async with (
        dataset_lock(dataset.id),
        set_database_global_context_variables(dataset.id, dataset.owner_id),
    ):
        purged: list = []  # (orphan, its deleted graph elements)
        for orphan in orphans:
            try:
                # Called unconditionally: gating on has_data_related_nodes would
                # skip graphs whose provenance is stamped in-graph instead of the
                # relational ledger (the function handles both paths and no-ops
                # when nothing is related).
                deleted_elements = await delete_data_nodes_and_edges(dataset.id, orphan.id, user.id)
                purged.append((orphan, deleted_elements))
            except Exception:
                failed.append(orphan.id)
                logger.warning(
                    "Failed to delete orphaned dlt row data_id=%s, skipping.",
                    orphan.id,
                    exc_info=True,
                )

        # One call for every purged row: their index nodes and routing rows go
        # in one round of store writes (per row, a sweep of thousands would
        # serialize thousands of commits under the dataset lock), then the
        # rows. A row whose delete fails keeps its Data row, so the next index
        # build rewrites what was removed for it.
        #
        # A row removed upstream is forgotten like forget(data_id=...): a user
        # decision whose chosen fact went with it is retired before the rows go.
        # One pass covers every purged row, since it reads what the graph holds
        # now; a failure keeps the rows for the next ingest's retry.
        retired: set[str] = set()
        if purged:
            try:
                retired = await retire_forgotten_decisions(dataset, user)
                await delete_data([orphan for orphan, _ in purged], dataset.id)
            except Exception:
                failed.extend(orphan.id for orphan, _ in purged)
                logger.warning(
                    "Failed to delete %d orphaned dlt row(s), skipping.",
                    len(purged),
                    exc_info=True,
                )

        # Invalidate session answers that used the orphans' graph elements or a
        # retired decision (best-effort; must not fail the cleanup). The graph
        # purge above is what they depend on, not the row delete.
        invalidations = [(deleted.node_ids, deleted.edge_ids) for _orphan, deleted in purged]
        if retired:
            invalidations.append((retired, set()))
        for node_ids, edge_ids in invalidations:
            try:
                from cognee.modules.session_lifecycle.invalidate_sessions import (
                    invalidate_sessions_for_deleted_data,
                )

                await invalidate_sessions_for_deleted_data(
                    dataset.id,
                    node_ids,
                    edge_ids,
                    user_id=user.id,
                )
            except Exception:
                logger.warning(
                    "Session invalidation after orphan cleanup failed (non-fatal).",
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
