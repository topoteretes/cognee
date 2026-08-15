"""Shared vector-store RE-KEY mechanics for data migrations.

STORE MECHANICS ONLY — no id derivations live here. Each migration freezes its
own id math (``_frozen_*``) locally; what is shared is the merge-safe way to
move a vector point from one id to another on each backend. These helpers are
part of every shipped migration's behavior: changes must stay backward-safe
(idempotent, merge-safe, tolerant of partial prior runs) for ALL migrations
that use them.

Native fast paths (LanceDB, PGVector) move rows carrying their stored vectors
— no re-embedding. Other backends fall back to re-embedding via
``index_data_points`` with the :class:`RekeyedPoint` carrier.
"""

import asyncio
import logging
import uuid as uuid_module

from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_context_config
from cognee.infrastructure.engine.models.DataPoint import DataPoint

logger = logging.getLogger(__name__)


async def index_data_points_batched(
    vector_engine, index_name: str, index_property_name: str, data_points: list
) -> None:
    """Re-embed and upsert ``data_points`` in embedding-engine batches.

    Mirrors cognify's own write path (``cognee/tasks/storage/index_data_points.py``):
    batches of ``embedding_engine.get_batch_size()`` so no single embedding request
    exceeds the provider's per-request token cap, with a semaphore bounding in-flight
    data points to ``embedding_max_concurrent_data_points``. Migrations must call
    this instead of ``vector_engine.index_data_points`` directly — one unbatched
    call over a whole collection is exactly what providers reject (OpenAI caps an
    embeddings request at 300k tokens total).
    """
    if not data_points:
        return

    batch_size = vector_engine.embedding_engine.get_batch_size()
    max_concurrent_data_points = get_embedding_context_config().embedding_max_concurrent_data_points
    semaphore = asyncio.Semaphore(max(1, max_concurrent_data_points // batch_size))

    async def _index_batch(batch):
        async with semaphore:
            await vector_engine.index_data_points(index_name, index_property_name, batch)

    batches = [data_points[i : i + batch_size] for i in range(0, len(data_points), batch_size)]

    # The FIRST batch runs alone: its write settles collection creation and any
    # lazy schema migration. Migrations write into tables the per-dataset
    # storage sync has not rebuilt yet, so on mismatch LanceDB drops and
    # recreates the table mid-write — a concurrent batch's lock-free
    # get_collection lands in that window and raises CollectionNotFoundError
    # (and every concurrent batch would redo the table rewrite). Once settled,
    # the remaining batches are plain merge-safe upserts and can overlap.
    await _index_batch(batches[0])

    if len(batches) > 1:
        await asyncio.gather(*(asyncio.create_task(_index_batch(batch)) for batch in batches[1:]))


class RekeyedPoint(DataPoint):
    """Carrier for re-inserting an existing vector point under a new id.

    Passed to ``vector_engine.index_data_points`` — the same write path cognify
    uses — so the adapter builds its own ``IndexSchema`` row ``{id, text,
    belongs_to_set}`` from it and the stored shape matches every existing row,
    on every vector backend.
    """

    text: str
    metadata: dict = {"index_fields": ["text"]}


def lancedb_where(ids: list[str]) -> str:
    """WHERE clause over ids, escaped the same way the adapter's own queries are."""
    escaped = [point_id.replace("'", "''") for point_id in ids]
    if len(escaped) == 1:
        return f"id = '{escaped[0]}'"
    return "id IN ({})".format(", ".join(f"'{point_id}'" for point_id in escaped))


async def rekey_lancedb(vector_engine, collection: str, id_map: dict) -> None:
    """Move LanceDB rows to new ids carrying their stored vectors (no re-embedding).

    Merge-safe and idempotent: an old id whose new id already exists in the
    table is deleted, never duplicated — so a crash between add and delete, a
    re-run, or a pre-existing new-scheme row all converge to one row per id.
    Uses plain ``add`` (append), NOT ``merge_insert`` (lance
    0.32 can panic with it on tables carrying deletion vectors), and compacts
    afterwards, best-effort (both local and subprocess-proxy table handles
    expose ``optimize``).
    """
    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        table = await vector_engine.get_collection(collection)
    except CollectionNotFoundError:
        return

    old_ids = [str(old_id) for old_id in id_map]
    rows = await table.query().where(lancedb_where(old_ids)).to_list()
    if not rows:
        return

    new_ids = [str(new_id) for new_id in id_map.values()]
    existing_new = {
        row["id"] for row in await table.query().where(lancedb_where(new_ids)).to_list()
    }

    moved_rows = []
    processed_old_ids = []
    for row in rows:
        new_id = str(id_map[str(row["id"])])
        processed_old_ids.append(str(row["id"]))
        if new_id in existing_new:
            # Target row already exists (equivalent content) -> merge by
            # dropping the old row instead of duplicating the id.
            continue
        new_row = dict(row)
        new_row["id"] = new_id
        payload = dict(new_row.get("payload") or {})
        payload["id"] = new_id
        new_row["payload"] = payload
        moved_rows.append(new_row)
        existing_new.add(new_id)  # two old ids -> one new id: move once, drop the rest

    if moved_rows:
        await table.add(moved_rows)
    await vector_engine.delete_data_points(collection, processed_old_ids)

    # Best-effort compaction: materializes the deletion vectors this re-key
    # just created (lance 0.32 reads/merge_inserts can panic on them). Uses a
    # FRESH handle (the deletes above advanced the table version, and optimize
    # from a stale handle raises a commit conflict); failure is non-fatal —
    # the data is already correct, only un-compacted.
    try:
        fresh_table = await vector_engine.get_collection(collection)
        optimize = getattr(fresh_table, "optimize", None)
        if optimize is not None:
            await optimize()
    except Exception as exc:  # noqa: BLE001 - compaction is an optimization
        logger.warning("Post-re-key compaction skipped for %s: %s", collection, exc)


async def rekey_pgvector(vector_engine, collection: str, id_map: dict) -> None:
    """Move PGVector rows to new primary-key ids with SQL UPDATEs — the vector
    column never moves, so nothing is re-embedded.

    Merge-safe and idempotent: an old id whose new id already exists is
    deleted instead of moved (the surviving row is equivalent), so re-runs and
    pre-existing new-scheme rows never raise a duplicate-key error.
    """
    from sqlalchemy import delete, select, update

    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        table = await vector_engine.get_table(collection)
    except CollectionNotFoundError:
        return

    async with vector_engine.get_async_session() as session:
        old_ids = [str(old_id) for old_id in id_map]
        new_ids = [str(new_id) for new_id in id_map.values()]

        existing_new = {
            str(row_id)
            for row_id in (
                await session.execute(select(table.c.id).where(table.c.id.in_(new_ids)))
            ).scalars()
        }
        rows = (
            await session.execute(
                select(table.c.id, table.c.payload).where(table.c.id.in_(old_ids))
            )
        ).all()

        for row in rows:
            new_id = str(id_map[str(row.id)])
            if new_id in existing_new:
                # Target row already exists -> merge by dropping the old row.
                await session.execute(delete(table).where(table.c.id == row.id))
                continue
            payload = dict(row.payload or {})
            payload["id"] = new_id
            await session.execute(
                update(table).where(table.c.id == row.id).values(id=new_id, payload=payload)
            )
            existing_new.add(new_id)  # two old ids -> one new id: move once
        await session.commit()


async def rekey_native(vector_engine, collection: str, id_map: dict) -> bool:
    """Vector-preserving re-key for backends that support it.

    Migration-local by design (no VectorDBInterface changes). Returns ``True``
    when the collection was handled natively (vectors moved, nothing
    re-embedded); ``False`` means the caller must use the generic re-embed
    path. Dispatch is by adapter class name so the optional backend packages
    are never imported here.
    """
    if not id_map:
        return True

    # NOT type(vector_engine): the engine arrives wrapped in the cache's
    # ``_LeasedValueProxy``, which spoofs ``__class__`` to the real adapter class
    # precisely so checks like this resolve through the wrapper.
    adapter = vector_engine.__class__.__name__
    if adapter == "LanceDBAdapter":
        await rekey_lancedb(vector_engine, collection, id_map)
        return True
    if adapter == "PGVectorAdapter":
        await rekey_pgvector(vector_engine, collection, id_map)
        return True
    return False


async def resync_document_id_native(vector_engine, collection: str, chunk_targets: dict) -> bool:
    """Repoint the ``document_id`` payload scalar on existing chunk vector rows
    WITHOUT re-embedding — the point id and chunk text are unchanged, so the
    stored vector is already correct; only the reference scalar moves.

    ``chunk_targets`` maps ``chunk point id -> new document id``. Returns ``True``
    when a native path handled it (nothing re-embedded); ``False`` means the
    caller must fall back to the generic re-embed path (``index_data_points``).

    Re-embedding purely to change a payload scalar is wasted embedding spend and,
    on a large fork, is exactly what trips a provider's per-request token cap.
    PGVector stores the payload as a plain JSON column, so it is updated in place.
    Backends whose vector rows cannot be updated without re-embedding (e.g.
    LanceDB's columnar store, whose payload arrow schema is not a plain map)
    return ``False`` and take the re-embed fallback.
    """
    if not chunk_targets:
        return True

    adapter = vector_engine.__class__.__name__
    if adapter == "PGVectorAdapter":
        await _resync_pgvector_document_id(vector_engine, collection, chunk_targets)
        return True
    return False


async def _resync_pgvector_document_id(vector_engine, collection: str, chunk_targets: dict) -> None:
    """PGVector: update the ``document_id`` payload key in place (vector untouched)."""
    from sqlalchemy import select, update

    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        table = await vector_engine.get_table(collection)
    except CollectionNotFoundError:
        return

    ids = [str(chunk_id) for chunk_id in chunk_targets]
    async with vector_engine.get_async_session() as session:
        rows = (
            await session.execute(select(table.c.id, table.c.payload).where(table.c.id.in_(ids)))
        ).all()
        for row in rows:
            new_document_id = str(chunk_targets[str(row.id)])
            payload = dict(row.payload or {})
            if payload.get("document_id") == new_document_id:
                continue  # idempotent: already re-keyed
            payload["document_id"] = new_document_id
            await session.execute(update(table).where(table.c.id == row.id).values(payload=payload))
        await session.commit()
