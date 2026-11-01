"""The classify and chunk tasks every standard cognify route starts with (SDK-397).

The indexing pipeline stores chunks before the cognify pipeline extracts their
entities, and the cognify pipeline skips re-embedding only the chunk ids it
finds in the vector store. Chunk ids are content-derived, so both runs must cut
exactly the same chunks: same chunker, same maximum size. Every task list that
chunks documents for those runs builds its chunking step here, so they cannot
drift apart.
"""

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents

# Chunks per downstream batch when neither the caller nor CHUNKS_PER_BATCH sets it.
DEFAULT_CHUNKS_PER_BATCH = 2000


def resolve_chunks_per_batch(chunks_per_batch: int | None = None) -> int:
    """The explicit value, else the CHUNKS_PER_BATCH setting, else the default."""
    if chunks_per_batch is not None:
        return chunks_per_batch
    configured = get_cognify_config().chunks_per_batch
    return configured if configured is not None else DEFAULT_CHUNKS_PER_BATCH


def build_chunking_tasks(chunker, max_chunk_size: int) -> tuple[Task, Task]:
    """(classify, chunk): classify raw Data items into Documents, then cut their chunks.

    ``max_chunk_size`` is the resolved token budget (``resolve_chunk_size``).
    Returned as a pair so a route can put its own task between them (the
    GLiNER route sketches its schema from the classified documents).
    """
    return (
        Task(classify_documents, needs_llm=False),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=max_chunk_size,
            chunker=chunker,
            needs_llm=False,
        ),
    )
