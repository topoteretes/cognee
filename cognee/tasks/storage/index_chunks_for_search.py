"""The indexing pipeline's write: make chunks searchable before extraction (SDK-397).

``cognify()`` runs two pipelines per dataset. The first, ``indexing_pipeline``
(``classify_documents`` → ``extract_chunks_from_documents`` → this task), writes
every document's chunks to the graph and vector stores as soon as they are cut,
so the chunk-based searches (``CHUNKS``, ``CHUNKS_LEXICAL``, ``RAG_COMPLETION``
and the chunk lane of ``HYBRID_COMPLETION``) answer while the second pipeline,
``cognify_pipeline``, is still extracting entities. That second pipeline
re-chunks the same text (chunk ids are content-derived, so it cuts the very
same chunks), asks the vector store which of them already have an embedding
(``find_indexed_chunks``) and upserts those without embedding them again.

The structural edges written here (``is_part_of``, ``belongs_to_set``) are not
indexed; the cognify pipeline's final write indexes them as it always did. The
vector written here is the one that stays, so a task that rewrites chunk text
(translation, redaction) belongs in both pipelines, before this task.

The ids embedded here are also recorded on ``ctx.extras[SEARCH_INDEXED_IDS]``,
which only matters when this task and ``add_data_points`` share one run (a
custom pipeline that chains them): the later write then skips them without
asking the store.
"""

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines.tasks.task import task_summary

from .add_data_points import SEARCH_INDEXED_IDS, add_data_points


@task_summary("Made {n} chunk(s) searchable")
async def index_chunks_for_search(
    data_chunks: list[DocumentChunk], ctx=None
) -> list[DocumentChunk]:
    """Store chunks (and the documents they belong to) in the graph and vector DBs.

    Args:
        data_chunks: The chunks one upstream batch produced. Each carries its
            document in ``is_part_of``; nothing else is attached yet.
        ctx: Pipeline runtime context, passed through to ``add_data_points``
            for provenance stamping. When present, the ids embedded here are
            recorded on ``ctx.extras[SEARCH_INDEXED_IDS]``.

    Returns:
        The same chunk objects, unchanged.
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
