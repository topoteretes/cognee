"""Find the chunks the indexing pipeline already embedded (SDK-397).

The cognify pipeline cuts the same chunks the indexing pipeline stored (chunk ids
are content-derived), so before its final write it asks the vector store which
of them already carry an embedding. Those ids go to
``ctx.extras[SEARCH_INDEXED_IDS]`` and ``add_data_points`` upserts their graph
rows without embedding them again. The documents those chunks belong to were
written in the same call as their chunks, so they are skipped too.

Verified against the store, not inferred from the indexing pipeline's completion
stamp: a stamp says "the item was indexed with some chunking", while the skip
must hold for exactly these chunk ids. After a chunk-size change, a rollback of
the index run, or ``forget(memory_only=True)`` the stamp and the store disagree,
and the store is right. One id lookup per few hundred chunks; a store that
cannot answer fails the run before extraction spends any tokens.

A chunk-size change can leave an earlier indexing run's chunks alongside the
new cut after extraction failed. Those earlier chunks are deleted before the
lookup (``retire_chunks_cut_differently``), even when some content IDs overlap
the new cut, so there is no second, entity-less copy of the text.
"""

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.methods.delete_chunks_incremental import (
    delete_chunks_incremental,
    edge_endpoints,
)
from cognee.modules.pipelines.models.DataItemStatus import is_item_stamped
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME

from .add_data_points import SEARCH_INDEXED_IDS

logger = get_logger("find_indexed_chunks")

# Ids per retrieve call: LanceDB answers one ``id IN (...)`` predicate per call.
LOOKUP_BATCH_SIZE = 500


async def _ids_in_store(vector_engine, collection_name: str, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(ids), LOOKUP_BATCH_SIZE):
        batch = ids[start : start + LOOKUP_BATCH_SIZE]
        # The adapters answer a missing collection (first run of a fresh store)
        # with no rows. Any other failure is the store being unreachable or
        # rejecting the query, and it propagates: extraction's final write goes
        # to the same store, so continuing would only pay the LLM for every
        # chunk before failing on it.
        rows = await vector_engine.retrieve(collection_name, batch)
        found.update(str(row.id) for row in rows)
    return found


# ctx.extras key: the item's document was checked for chunks cut differently.
STALE_CHUNKS_CHECKED = "stale_chunks_checked"


def _cut_differently(node: dict, chunk: DocumentChunk) -> bool:
    """Whether a stored chunk was cut with other settings than ``chunk``.

    Only recorded settings count: a chunk without them (written before they
    were recorded) is never taken for a stale one.
    """
    stored_tokens = node.get("max_chunk_tokens")
    stored_chunker = node.get("chunker_id")
    if (
        stored_tokens is not None
        and chunk.max_chunk_tokens is not None
        and int(stored_tokens) != int(chunk.max_chunk_tokens)
    ):
        return True
    return bool(stored_chunker and chunk.chunker_id and stored_chunker != chunk.chunker_id)


async def retire_chunks_cut_differently(data_chunks: list[DocumentChunk], ctx) -> list[str]:
    """Delete the document's chunks an earlier indexing run cut with other settings.

    Runs once per item, and only for an item an earlier indexing run stamped
    whose extraction never completed: its stored chunks came from that run
    alone, and the extraction they waited for is this one. Chunk-scoped
    deletion needs ownership kept in the graph; elsewhere nothing is deleted.
    A failure fails the run: continuing would leave the old cut next to the
    new one, so chunk searches return the passage twice, once without
    entities. Extraction's own write would fail on the same graph store.
    """
    data_item = getattr(ctx, "data_item", None)
    dataset = getattr(ctx, "dataset", None)
    document = getattr(data_chunks[0], "is_part_of", None)
    if data_item is None or dataset is None or document is None:
        return []
    checked = ctx.extras.setdefault(STALE_CHUNKS_CHECKED, set())
    if str(document.id) in checked:
        return []
    checked.add(str(document.id))
    if not is_item_stamped(data_item, INDEXING_PIPELINE_NAME, dataset.id) or is_item_stamped(
        data_item, COGNIFY_PIPELINE_NAME, dataset.id
    ):
        return []
    graph_engine = await get_graph_engine()
    if not await stores_provenance_in_graph(graph_engine):
        return []
    chunk_ids = []
    for source, edge, target in await graph_engine.get_connections(str(document.id)):
        if "is_part_of" not in str(edge.get("relationship_name", "")):
            continue
        source_id, target_id = edge_endpoints(source, edge, target)
        if target_id == str(document.id) and source_id != str(document.id):
            chunk_ids.append(source_id)
    nodes = await graph_engine.get_nodes(chunk_ids) if chunk_ids else []
    stale = [
        str(node["id"])
        for node in nodes
        if node.get("type") == type(data_chunks[0]).__name__
        and _cut_differently(node, data_chunks[0])
    ]
    if stale:
        await delete_chunks_incremental(stale, dataset.id, data_item.id)
        logger.info(
            "Deleted %d chunk(s) of %r that an earlier indexing run cut with other "
            "chunking settings; the document is indexed anew.",
            len(stale),
            getattr(document, "name", str(document.id)),
        )
    return stale


@task_summary("Checked {n} chunk(s) for an existing index")
async def find_indexed_chunks(data_chunks: list[DocumentChunk], ctx=None) -> list[DocumentChunk]:
    """Record on ``ctx`` which of ``data_chunks`` the vector store already holds.

    Returns the same chunk objects, unchanged, for the extraction task that
    follows. Without a context there is nowhere to record the ids, so the
    final write embeds everything, exactly as it does with the indexing pipeline
    turned off.
    """
    if not data_chunks or ctx is None:
        return data_chunks

    # A short chunk can retain its content-derived ID across a chunk-size
    # change. Check the old cut even when some IDs overlap, before looking up
    # vectors: retirement can delete one of those shared IDs too.
    await retire_chunks_cut_differently(data_chunks, ctx)

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
