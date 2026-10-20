"""Find the chunks the index pipeline already embedded (SDK-397).

The cognify pipeline cuts the same chunks the index pipeline stored (chunk ids
are content-derived), so before its final write it asks the vector store which
of them already carry an embedding. Those ids go to
``ctx.extras[SEARCH_INDEXED_IDS]`` and ``add_data_points`` upserts their graph
rows without embedding them again. The documents those chunks belong to were
written in the same call as their chunks, so they are skipped too.

Verified against the store, not inferred from the index pipeline's completion
stamp: a stamp says "the item was indexed with some chunking", while the skip
must hold for exactly these chunk ids. After a chunk-size change, a rollback of
the index run, or ``forget(memory_only=True)`` the stamp and the store disagree,
and the store is right. One id lookup per few hundred chunks; a store that
cannot answer counts as holding nothing, because embedding a chunk twice costs
tokens while skipping one that is not there would leave it unsearchable.
"""

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger

from .add_data_points import SEARCH_INDEXED_IDS

logger = get_logger("find_indexed_chunks")

# Ids per retrieve call: LanceDB answers one ``id IN (...)`` predicate per call.
LOOKUP_BATCH_SIZE = 500


async def _ids_in_store(vector_engine, collection_name: str, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(ids), LOOKUP_BATCH_SIZE):
        batch = ids[start : start + LOOKUP_BATCH_SIZE]
        try:
            rows = await vector_engine.retrieve(collection_name, batch)
        except Exception as error:  # noqa: BLE001 - an unreadable store means re-embedding, not failing
            # Includes the collection not existing yet (first run of a fresh
            # store): nothing is indexed, the final write embeds everything.
            logger.info(
                "Could not read %s for already indexed chunks (%s: %s); "
                "the final write embeds this batch.",
                collection_name,
                type(error).__name__,
                error,
            )
            return found
        found.update(str(row.id) for row in rows)
    return found


@task_summary("Checked {n} chunk(s) for an existing index")
async def find_indexed_chunks(data_chunks: list[DocumentChunk], ctx=None) -> list[DocumentChunk]:
    """Record on ``ctx`` which of ``data_chunks`` the vector store already holds.

    Returns the same chunk objects, unchanged, for the extraction task that
    follows. Without a context there is nowhere to record the ids, so the
    final write embeds everything, exactly as it does with the index pipeline
    turned off.
    """
    if not data_chunks or ctx is None:
        return data_chunks

    # Chunk text lives in the ``<ChunkType>_text`` collection (index_data_points
    # names collections ``<type>_<field>``); group by type in case a custom
    # chunker subclasses DocumentChunk.
    ids_by_collection: dict[str, list[str]] = {}
    for chunk in data_chunks:
        ids_by_collection.setdefault(f"{type(chunk).__name__}_text", []).append(str(chunk.id))

    vector_engine = await get_vector_engine_async()
    found_ids: set[str] = set()
    for collection_name, ids in ids_by_collection.items():
        found_ids |= await _ids_in_store(vector_engine, collection_name, ids)

    if not found_ids:
        return data_chunks

    indexed = ctx.extras.setdefault(SEARCH_INDEXED_IDS, set())
    for chunk in data_chunks:
        if str(chunk.id) not in found_ids:
            continue
        indexed.add(str(chunk.id))
        document = getattr(chunk, "is_part_of", None)
        if document is not None:
            indexed.add(str(document.id))

    logger.info(
        "%d of %d chunk(s) are already indexed and will not be embedded again.",
        len(found_ids),
        len(data_chunks),
    )
    return data_chunks
