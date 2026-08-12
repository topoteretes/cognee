"""Row-level update for a DLT source manifest (surgical delta apply).

update() hands us both sides of the diff in one call: the old identity (the
manifest Data record, by UUID) and the new source content. That removes the
discovery problem the cognify purge-and-rebuild path has to solve — the delta
is a set operation on content-addressed row node ids (table:pk:content_hash):

    removed   = old − new   → delete row nodes (edges detach) and row vectors
    added     = new − old   → chunk, embed, and wire schema/FK edges
    unchanged = old ∩ new   → untouched; FK edges re-emitted only when the
                              referenced row's node id changed under them

An *edited* row is a removed+added pair under the same (table, pk) — its
content hash, and therefore its node id, changed.

Shared nodes (SchemaTable, SchemaRelationship, DltColumn) are never deleted
here: they are deterministic-id upserts owned by no single row. Ledger entries
for removed nodes are left in place — a later full purge deleting an already
absent node is a no-op.

After the delta is applied, the manifest's cognify_pipeline status is set back
to COMPLETED for this dataset: the derived stores are in sync with the new
manifest, so a later cognify() must skip it instead of running the
purge-and-rebuild route over work this path already did.
"""

import json
from os.path import basename
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.shared.logging_utils import get_logger

from .create_dlt_source import (
    create_dlt_source_from_connection_string,
    create_dlt_source_from_csv,
    is_connection_string,
    is_csv_path,
)
from .dlt_utils import load_dlt_manifest

logger = get_logger()

DLT_ROW_VECTOR_COLLECTION = "DltRow_text"


def _single_dlt_item(data):
    """Normalize update() input to exactly one dlt item, or None if not DLT.

    update() targets ONE Data record, so a list input must contain exactly one
    source. CSV paths and connection strings are auto-converted the same way
    resolve_dlt_sources does.
    """
    try:
        from dlt.extract import DltResource, SourceFactory
        from dlt.extract.source import DltSource
    except ImportError:
        return None

    items = data if isinstance(data, list) else [data]
    if len(items) != 1:
        return None
    item = items[0]

    if isinstance(item, str) and is_csv_path(item):
        return create_dlt_source_from_csv(item)
    if isinstance(item, str) and is_connection_string(item):
        return create_dlt_source_from_connection_string(item)
    if isinstance(item, (DltResource, DltSource, SourceFactory)):
        return item
    return None


async def _verify_manifest_identity(dlt_item, data_record, dataset_name, user) -> str:
    """The new source must resolve to the SAME manifest identity as data_id.

    The manifest id is seeded from (dataset, source name); a source with a
    different name would create a second manifest and leave the target record
    untouched — fail loudly before any ingestion happens.
    """
    from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id

    source_name = getattr(dlt_item, "name", None) or dataset_name
    expected_id = await get_unique_data_id(f"dlt_source:{dataset_name}:{source_name}", user)
    if expected_id != data_record.id:
        raise ValueError(
            f"The provided source (name {source_name!r}) resolves to manifest id "
            f"{expected_id}, but the update targets {data_record.id}. A DLT update "
            "must supply a source with the same name as the one being updated — "
            "renaming a source is remove + add, not update."
        )
    return source_name


def _rows_by_node_id(manifest: dict) -> dict[str, dict]:
    return {row["node_id"]: row for row in manifest.get("rows", [])}


async def update_dlt_source_rows(
    data_record,
    data,
    dataset,
    user,
    primary_key=None,
    write_disposition: str = "replace",
) -> dict:
    """Apply a new version of a DLT source as a row-level delta.

    Returns a summary dict with the row counts per delta category.
    """
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.modules.data.methods.get_data import get_data

    dlt_item = _single_dlt_item(data)
    if dlt_item is None:
        raise ValueError(
            "DLT manifest update requires exactly one DLT source as data "
            "(a dlt resource/source, a CSV path, or a connection string)."
        )
    await _verify_manifest_identity(dlt_item, data_record, dataset.name, user)

    old_manifest = await load_dlt_manifest(data_record.raw_data_location)
    old_rows = _rows_by_node_id(old_manifest)

    # Re-ingest through add(): staging load (dlt's own merge/replace/append
    # machinery), manifest rebuild, and the in-place Data record update all
    # reuse the existing machinery. This clears the record's pipeline_status
    # (content changed) — restored below once the delta is applied.
    from cognee.api.v1.add import add  # lazy: avoids api <-> tasks import cycle

    # dataset_name must be explicit: resolve_dlt_sources runs before add()'s
    # dataset resolution, and its default ("main_dataset") would compute a
    # different manifest identity than the record being updated.
    await add(
        data=[dlt_item],
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        user=user,
        write_disposition=write_disposition,
        primary_key=primary_key,
    )

    fresh_record = await get_data(user.id, data_record.id)
    if fresh_record is None:
        raise RuntimeError(f"Manifest record {data_record.id} disappeared during update.")
    new_manifest = await load_dlt_manifest(fresh_record.raw_data_location)
    new_rows = _rows_by_node_id(new_manifest)

    removed_ids = old_rows.keys() - new_rows.keys()
    added_ids = new_rows.keys() - old_rows.keys()
    unchanged_ids = old_rows.keys() & new_rows.keys()

    # Edited rows = same (table, pk) on both sides with a different node id.
    old_pks = {(r["table_name"], r["primary_key_value"]) for r in old_rows.values()}
    edited_pks = {
        (r["table_name"], r["primary_key_value"])
        for nid, r in new_rows.items()
        if nid in added_ids and (r["table_name"], r["primary_key_value"]) in old_pks
    }

    # Unchanged rows whose FK targets changed identity (the referenced row was
    # edited) need their FK edges re-emitted; the old edges vanish when the old
    # target node is deleted (detach), so only re-emission is required.
    fk_repair_ids = {
        nid
        for nid in unchanged_ids
        if old_rows[nid].get("fk_references") != new_rows[nid].get("fk_references")
    }

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()

        if removed_ids:
            await graph_engine.delete_nodes([str(node_id) for node_id in removed_ids])
            vector_engine = await get_vector_engine_async()
            await vector_engine.delete_data_points(
                DLT_ROW_VECTOR_COLLECTION, [UUID(node_id) for node_id in removed_ids]
            )

        ctx = PipelineContext(
            user=user,
            dataset=dataset,
            data_item=fresh_record,
            pipeline_name="cognify_pipeline",
        )

        if added_ids:
            await _add_rows(fresh_record, new_manifest, added_ids, ctx)

        emit_ids = added_ids | fk_repair_ids
        if emit_ids:
            from .dlt_schema_graph import emit_dlt_schema_graph

            records = [
                {
                    "source_id": nid,
                    "table_name": new_rows[nid]["table_name"],
                    "fk_references": new_rows[nid].get("fk_references", []),
                    "column_values": new_rows[nid].get("column_values", {}),
                }
                for nid in sorted(emit_ids)
            ]
            await emit_dlt_schema_graph(new_manifest["tables"], records, ctx=ctx)

    await _mark_cognified(fresh_record.id, dataset.id)

    summary = {
        "rows_added": len(added_ids) - len(edited_pks),
        "rows_edited": len(edited_pks),
        "rows_removed": len(removed_ids) - len(edited_pks),
        "rows_unchanged": len(unchanged_ids),
        "fk_edges_repaired": len(fk_repair_ids),
    }
    logger.info("DLT row-level update applied for manifest %s: %s", fresh_record.id, summary)
    return summary


async def _add_rows(data_record, manifest: dict, node_ids: set[str], ctx: PipelineContext):
    """Chunk, embed, and store the added rows (mirrors DltSourceDocument.read)."""
    from cognee.modules.chunking.models.DltRow import DltRow
    from cognee.modules.cognify.config import get_cognify_config
    from cognee.tasks.documents.classify_documents import classify_documents
    from cognee.tasks.storage.add_data_points import add_data_points

    document = (await classify_documents([data_record]))[0]

    chunks = []
    for chunk_index, row in enumerate(manifest.get("rows", [])):
        if row["node_id"] not in node_ids:
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        chunks.append(
            DltRow(
                id=UUID(row["node_id"]),
                text=text,
                chunk_size=len(text.split()),
                chunk_index=chunk_index,
                cut_type="dlt_row",
                is_part_of=document,
                contains=[],
                document_id=str(document.id),
                document_name=document.name or basename(document.raw_data_location),
            )
        )

    if chunks:
        cognify_config = get_cognify_config()
        await add_data_points(chunks, ctx=ctx, embed_triplets=cognify_config.triplet_embedding)


async def _mark_cognified(data_id: UUID, dataset_id: UUID):
    """Mark the manifest cognified for this dataset (mirrors run_tasks_data_item).

    add() cleared pipeline_status because the content changed; the delta apply
    has brought the derived stores in sync, so a later cognify() must skip this
    item rather than purge-and-rebuild it.
    """
    from sqlalchemy import select

    from cognee.modules.data.models import Data

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        if data_point is None:
            raise RuntimeError(f"Data record {data_id} not found while finalizing update.")
        status_for_pipeline = data_point.pipeline_status.setdefault("cognify_pipeline", {})
        status_for_pipeline[str(dataset_id)] = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
        await session.merge(data_point)
        await session.commit()
