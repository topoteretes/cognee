"""The classify and chunk tasks every standard cognify route starts with (SDK-397).

The indexing pipeline stores chunks before the cognify pipeline extracts their
entities, and the cognify pipeline skips re-embedding only the chunk ids it
finds in the vector store. Chunk ids are content-derived, so both runs must cut
exactly the same chunks: same chunker, same maximum size. Every task list that
chunks documents for those runs builds its chunking step here, so they cannot
drift apart. The step ends with the tasks plugins register to run on the chunks
(``chunk_task_registry.py``), so both runs also pass them the same chunks.
"""

from cognee.modules.cognify.chunk_task_registry import chunk_task_factories
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


def build_registered_chunk_tasks(chunks_per_batch: int) -> list[Task]:
    """The tasks the registered factories return, in registration order.

    ``chunks_per_batch`` is the batch size of the tasks after the chunking
    step; every factory receives it. A factory that returns anything but a
    ``Task`` or a list of them raises ``TypeError``.
    """
    tasks = []
    for name, factory in chunk_task_factories():
        built = factory(chunks_per_batch=chunks_per_batch)
        factory_tasks = [built] if isinstance(built, Task) else built
        if not isinstance(factory_tasks, (list, tuple)) or not all(
            isinstance(task, Task) for task in factory_tasks
        ):
            raise TypeError(
                f"Chunk task factory {name!r} returned {type(built).__name__}; "
                "expected a Task or a list of Tasks."
            )
        tasks.extend(factory_tasks)
    return tasks


def build_chunking_tasks(
    chunker,
    max_chunk_size: int,
    temporal_hints: bool = True,
    chunks_per_batch: int | None = None,
) -> tuple[Task, ...]:
    """(classify, chunk, *registered): classify raw Data items into Documents, cut
    their chunks, then run the tasks registered with ``register_chunk_task``.

    ``max_chunk_size`` is the resolved token budget (``resolve_chunk_size``).
    Classification comes first so a route can put its own task right after it
    (the GLiNER route sketches its schema from the classified documents).
    ``temporal_hints`` is off for the indexing pipeline: the hints only feed
    graph extraction, which the indexing run never reaches, and they do not
    change the chunk text, so both runs still cut the same chunk ids.
    ``chunks_per_batch`` (resolved by ``resolve_chunks_per_batch``) is the
    batch size of the list's later tasks, handed to the registered factories.
    With nothing registered this is the (classify, chunk) pair.
    """
    return (
        Task(classify_documents, needs_llm=False),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=max_chunk_size,
            chunker=chunker,
            temporal_hints=temporal_hints,
            needs_llm=False,
        ),
        *build_registered_chunk_tasks(resolve_chunks_per_batch(chunks_per_batch)),
    )
