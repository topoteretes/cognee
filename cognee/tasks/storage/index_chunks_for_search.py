"""Make a dataset searchable before its entities are extracted (SDK-397).

Entity extraction is the slow step of ``cognify()``: an LLM (or GLiNER) pass
per chunk batch. Until it finishes, every search on the dataset finds nothing,
although the chunk texts themselves were ready the moment chunking ended. This
task closes that gap by writing the chunks (and the documents they belong to)
to the graph and vector stores right after chunking, so the chunk-based
searches (``CHUNKS``, ``CHUNKS_LEXICAL``, ``RAG_COMPLETION`` and the chunk lane
of ``HYBRID_COMPLETION``) answer while extraction is still running. The
entity-aware searches catch up when ``add_data_points`` runs at the end of the
pipeline. Both extractor routes of ``cognify()`` run it; ``update()``'s
incremental path does not (it re-extracts only edited regions).

The final ``add_data_points`` receives the same chunks again, now carrying
their entities and summaries. Their graph rows are upserted a second time
(idempotent, and that pass is what attaches the entities), but the chunk text
is not embedded twice: the ids embedded here are recorded on
``ctx.extras[SEARCH_INDEXED_IDS]`` and ``add_data_points`` skips them. The
structural edges written here (``is_part_of``, ``belongs_to_set``) are not
indexed either; the final write indexes them as it always did.

That skip is by id, so the vector written here is the one that stays: a task
that rewrites chunk text (translation, redaction) must run BEFORE this task,
or its chunks would stay searchable under the original text.
"""

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines.tasks.task import task_summary

from .add_data_points import SEARCH_INDEXED_IDS, add_data_points


@task_summary("Made {n} chunk(s) searchable")
async def index_chunks_for_search(
    data_chunks: list[DocumentChunk], ctx=None
) -> list[DocumentChunk]:
    """Store chunks in the graph and vector DBs ahead of entity extraction.

    Args:
        data_chunks: The chunks one upstream batch produced. Each carries its
            document in ``is_part_of``; nothing else is attached yet.
        ctx: Pipeline runtime context. When present, the ids of every node
            embedded here are recorded on ``ctx.extras[SEARCH_INDEXED_IDS]`` so
            the final ``add_data_points`` does not embed them again. Without a
            context the write still happens; the later pass then re-embeds.

    Returns:
        The same chunk objects, unchanged, for the extraction task that follows.
    """
    if not data_chunks:
        return data_chunks

    await add_data_points(list(data_chunks), ctx=ctx, index_edges=False)

    if ctx is not None:
        indexed = ctx.extras.setdefault(SEARCH_INDEXED_IDS, set())
        for chunk in data_chunks:
            indexed.add(str(chunk.id))
            document = getattr(chunk, "is_part_of", None)
            if document is not None:
                indexed.add(str(document.id))

    return data_chunks
