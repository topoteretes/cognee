"""Run the caching, indexing and cognify pipelines in order (SDK-397).

``cognify()`` builds memory in three pipelines per dataset, each a run of its
own (its own ``pipeline_runs`` row and name, per-item completion stamps,
telemetry and rollback), so a dataset answers queries as soon as possible and
the answers improve at each stage:

1. ``caching_pipeline`` — store each document's text. No chunking, no
   embedding, no LLM. When it completes the dataset answers keyword (BM25)
   searches: ``DOCUMENTS_LEXICAL``. Memory state ``cached``.
2. ``indexing_pipeline`` — chunk, store and embed the chunks. When it
   completes the chunk searches answer (``CHUNKS``, ``CHUNKS_LEXICAL``,
   ``RAG_COMPLETION``, ``HYBRID_COMPLETION``). Memory state ``indexed``.
3. ``cognify_pipeline`` — extract entities and relationships, summarize, write
   the graph. It cuts the same chunks stage 2 stored, finds them again in the
   vector store (``find_indexed_chunks``) and does not embed them twice.
   Memory state ``ready``.

:func:`run_memory_pipelines` chains the runs so that one executor call,
blocking or background, drives all of them in order; each run takes the
per-dataset lock on its own, exactly as ``run_pipeline`` always has. It yields
every run info of every run, so the blocking executor's result is still the
cognify run's final info, and the background executor hands back the first
run's started info (its subscribers keep receiving the later runs' events,
see ``run_pipeline_as_background_process``). A failed preparation run never
decides whether the next one runs: a dataset whose caching or indexing failed
is still extracted, and the final write stores whatever the earlier runs did
not.

Each milestone is logged, so the memory's state can be read from the log
stream alone, and stamped on the event as ``memory_state`` (the vocabulary of
``cognee.modules.pipelines.operations.get_memory_status``).
"""

from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunInfo,
    PipelineRunStageCompleted,
    PipelineRunStarted,
)
from cognee.modules.pipelines.operations.get_memory_status import (
    MEMORY_CACHED,
    MEMORY_CACHING_FAILED,
    MEMORY_EXTRACTION_FAILED,
    MEMORY_INDEXED,
    MEMORY_INDEXING_FAILED,
    MEMORY_READY,
)
from cognee.modules.pipelines.operations.pipeline import run_pipeline
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import (
    CACHING_PIPELINE_NAME,
    COGNIFY_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
)

logger = get_logger("cognify")

# run_pipeline parameters the preparation runs share with the cognify run.
# The task list, the pipeline name, the LLM need and the after-run hooks are
# per run; ``extras`` carries cognify's resolved extractor, which the earlier
# runs have no use for.
_SHARED_RUN_PARAMS = (
    "datasets",
    "user",
    "vector_db_config",
    "graph_db_config",
    "incremental_loading",
    "data_per_batch",
    "rollback_handler",
    "llm_config",
    "embedding_config",
    "data_cache",
    "skip_connection_test",
)


@dataclass(frozen=True)
class MemoryStage:
    """One preparation run and the memory state it leads to."""

    pipeline_name: str
    tasks: Any  # a task list or a per-item resolver, as run_pipeline takes it
    completed_state: str
    failed_state: str
    purpose: str  # what the run does, for the started line
    completed_detail: str  # what answers once it completed
    failed_detail: str  # what the later runs still do


CACHING_STAGE = MemoryStage(
    pipeline_name=CACHING_PIPELINE_NAME,
    tasks=None,
    completed_state=MEMORY_CACHED,
    failed_state=MEMORY_CACHING_FAILED,
    purpose="storing the documents' text for keyword (BM25) search before chunking and extraction",
    completed_detail=(
        "keyword searches (DOCUMENTS_LEXICAL) answer now; chunking and embedding next, "
        "then entity extraction, and answers improve at each step"
    ),
    failed_detail=(
        "chunking and extraction run anyway; nothing new answers keyword searches "
        "until a later run caches these documents"
    ),
)

INDEXING_STAGE = MemoryStage(
    pipeline_name=INDEXING_PIPELINE_NAME,
    tasks=None,
    completed_state=MEMORY_INDEXED,
    failed_state=MEMORY_INDEXING_FAILED,
    purpose="chunking and embedding so chunk searches answer before entity extraction",
    completed_detail=(
        "chunk searches (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, HYBRID_COMPLETION) answer "
        "now; entity extraction next, and graph-based answers improve once it completes"
    ),
    failed_detail=(
        "entity extraction runs anyway and its final write stores the chunks itself, "
        "so the dataset becomes searchable at chunk level when it completes"
    ),
)

# What a dataset still answers after a failed extraction, by the best state
# an earlier run reached for it.
_STILL_ANSWERS = {
    MEMORY_INDEXED: "its chunks stay searchable (CHUNKS, RAG_COMPLETION, HYBRID_COMPLETION)",
    MEMORY_CACHED: "its documents stay searchable by keyword (DOCUMENTS_LEXICAL)",
}


def _item_counts(run_info: PipelineRunInfo) -> tuple[int, int]:
    """(items processed, items skipped as already done) of a completed run."""
    processed = skipped = 0
    for entry in run_info.data_ingestion_info or []:
        item_info = entry.get("run_info") if isinstance(entry, dict) else None
        if isinstance(item_info, PipelineRunAlreadyCompleted):
            skipped += 1
        elif isinstance(item_info, PipelineRunCompleted):
            processed += 1
    return processed, skipped


def _failure(run_info: PipelineRunErrored) -> str:
    cause = run_info.error_class or "unknown error"
    if run_info.error_message:
        cause = f"{cause}: {run_info.error_message}"
    return cause


async def _run_stage(
    stage: MemoryStage, run_params: dict[str, Any], reached: dict[UUID, str]
) -> AsyncIterator[PipelineRunInfo]:
    """Yield one preparation run's events, logging and stamping its milestones."""
    stage_run = run_pipeline(
        tasks=stage.tasks,
        pipeline_name=stage.pipeline_name,
        needs_llm=False,
        **run_params,
    )
    try:
        async with aclosing(stage_run):
            async for run_info in stage_run:
                if isinstance(run_info, PipelineRunStarted):
                    logger.info(
                        "Preparing dataset %r (%s, run %s): %s.",
                        run_info.dataset_name,
                        stage.pipeline_name,
                        run_info.pipeline_run_id,
                        stage.purpose,
                    )
                elif isinstance(run_info, PipelineRunCompleted):
                    reached[run_info.dataset_id] = stage.completed_state
                    processed, skipped = _item_counts(run_info)
                    logger.info(
                        "Dataset %r is %s (memory state: %s): %d item(s) processed, %d already "
                        "were. From now on %s.",
                        run_info.dataset_name,
                        stage.completed_state,
                        stage.completed_state,
                        processed,
                        skipped,
                        stage.completed_detail,
                    )
                    run_info = PipelineRunStageCompleted(
                        pipeline_run_id=run_info.pipeline_run_id,
                        dataset_id=run_info.dataset_id,
                        dataset_name=run_info.dataset_name,
                        payload=run_info.payload,
                        data_ingestion_info=run_info.data_ingestion_info,
                        pipeline_name=stage.pipeline_name,
                        memory_state=stage.completed_state,
                    )
                elif isinstance(run_info, PipelineRunErrored):
                    run_info.memory_state = stage.failed_state
                    logger.warning(
                        "%s for dataset %r failed (%s; memory state: %s): %s.",
                        stage.pipeline_name,
                        run_info.dataset_name,
                        _failure(run_info),
                        stage.failed_state,
                        stage.failed_detail,
                    )
                yield run_info
    except Exception as error:  # noqa: BLE001 - a failed stage never blocks the next one
        # A run-level failure (setup, authorization, an exception escaping a
        # task). The next run gets its own attempt; it fails the same way if
        # the cause persists, and that failure is the one callers see.
        logger.warning(
            "%s raised %s: %s. Continuing with the next memory pipeline.",
            stage.pipeline_name,
            type(error).__name__,
            error,
        )


async def run_memory_pipelines(
    caching_tasks: list | Callable[[Any], list] | None = None,
    indexing_tasks: list | Callable[[Any], list] | None = None,
    **cognify_params: Any,
) -> AsyncIterator[PipelineRunInfo]:
    """Yield the caching run's events, then the indexing run's, then the cognify run's.

    ``cognify_params`` are ``run_pipeline``'s keyword arguments for the cognify
    run (``tasks``, ``pipeline_name="cognify_pipeline"``, hooks, ...).
    ``caching_tasks`` and ``indexing_tasks`` are the earlier runs' task lists or
    per-item resolvers; ``None`` skips that run (``document_caching=False``,
    ``early_chunk_indexing=False``).

    A preparation run's ``PipelineRunCompleted`` is yielded as
    :class:`PipelineRunStageCompleted`, so a consumer that stops at "completed"
    keeps following into the run that comes next.
    """
    run_params = {k: v for k, v in cognify_params.items() if k in _SHARED_RUN_PARAMS}
    stages = []
    if caching_tasks is not None:
        stages.append(MemoryStage(**{**CACHING_STAGE.__dict__, "tasks": caching_tasks}))
    if indexing_tasks is not None:
        stages.append(MemoryStage(**{**INDEXING_STAGE.__dict__, "tasks": indexing_tasks}))

    # The run each dataset's chain started with: a caller that was handed that
    # run id (background mode) keeps receiving the later runs' events under it.
    first_run_ids: dict[UUID, UUID] = {}

    def _mark_chain(run_info: PipelineRunInfo) -> PipelineRunInfo:
        first = first_run_ids.setdefault(run_info.dataset_id, run_info.pipeline_run_id)
        if first != run_info.pipeline_run_id:
            run_info.chain_run_id = first
        return run_info

    # The best state each dataset reached before extraction, for the failure line.
    reached: dict[UUID, str] = {}
    for stage in stages:
        async for run_info in _run_stage(stage, run_params, reached):
            yield _mark_chain(run_info)

    pipeline_name = cognify_params.get("pipeline_name", COGNIFY_PIPELINE_NAME)
    cognify_run = run_pipeline(**cognify_params)
    async with aclosing(cognify_run):
        async for run_info in cognify_run:
            if isinstance(run_info, PipelineRunStarted):
                logger.info(
                    "Extracting entities and relationships for dataset %r (%s, run %s).",
                    run_info.dataset_name,
                    pipeline_name,
                    run_info.pipeline_run_id,
                )
            elif isinstance(run_info, PipelineRunCompleted):
                run_info.memory_state = MEMORY_READY
                processed, skipped = _item_counts(run_info)
                logger.info(
                    "Dataset %r memory is fully built (memory state: %s): entities and "
                    "relationships extracted for %d item(s), %d already were.",
                    run_info.dataset_name,
                    MEMORY_READY,
                    processed,
                    skipped,
                )
            elif isinstance(run_info, PipelineRunErrored):
                run_info.memory_state = MEMORY_EXTRACTION_FAILED
                still = _STILL_ANSWERS.get(
                    reached.get(run_info.dataset_id), "nothing new is searchable"
                )
                logger.warning(
                    "Entity extraction for dataset %r failed (%s; memory state: %s); %s. "
                    "Fix the cause and re-run cognify() to build the graph.",
                    run_info.dataset_name,
                    _failure(run_info),
                    MEMORY_EXTRACTION_FAILED,
                    still,
                )
            yield _mark_chain(run_info)
