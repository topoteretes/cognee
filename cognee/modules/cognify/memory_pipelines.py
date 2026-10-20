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
blocking or background, drives all of them in order; a dataset's runs hold
its lock as one unit, as a single cognify run did. The cognify run starts
first and pauses after its start event (``_dataset_chain`` says why), so the
background executor hands back the cognify run's id, as it always has, and
its subscribers keep receiving the preparation runs' events too (see
``run_pipeline_as_background_process``); the blocking executor's result is
still the cognify run's final info. The preparation runs only process the
documents that still need them: not DLT manifests or code files, and with
incremental loading not documents already extracted. A failed preparation
run never decides whether extraction runs: a dataset whose caching or
indexing failed is still extracted, and the final write stores whatever the
earlier runs did not.

Each milestone is logged, so the memory's state can be read from the log
stream alone, and stamped on the event as ``memory_state`` (the vocabulary of
``cognee.modules.pipelines.operations.get_memory_status``).
"""

from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cognee.infrastructure.locks import get_dataset_lock, held_datasets
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.setup_and_check_environment import (
    setup_and_check_environment,
)
from cognee.modules.pipelines.models.DataItemStatus import is_data_item_completed
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
from cognee.modules.pipelines.operations.pipeline import drive_marking_held, run_pipeline
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import (
    CACHING_PIPELINE_NAME,
    COGNIFY_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
)

logger = get_logger("cognify")

# run_pipeline parameters the preparation runs share with the cognify run.
# The task list, the pipeline name, the LLM need and the after-run hooks are
# per run; cognify's ``extras`` carry its resolved extractor, which the earlier
# runs have no use for (theirs carry the chain's run id).
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
    except Exception as error:
        # A run-level failure (setup, authorization, an exception escaping a
        # task). The next run gets its own attempt; it fails the same way if
        # the cause persists, and that failure is the one callers see. A
        # failed stage never blocks the next one, so this handler is broad on
        # purpose, and logs the traceback.
        logger.warning(
            "%s raised %s: %s. Continuing with the next memory pipeline.",
            stage.pipeline_name,
            type(error).__name__,
            error,
            exc_info=True,
        )


def _stage_tasks_for(stage: MemoryStage, item: Any):
    """The task list ``stage`` runs for ``item`` (its route's list, or the stage's list)."""
    if not callable(stage.tasks):
        return stage.tasks
    try:
        return stage.tasks(item)
    except (ValueError, KeyError):
        # An item routing rejects (a legacy DLT row, an unmapped route) gets no
        # preparation; the cognify run reports the routing error for it.
        return None


def _is_stamped(item: Any, pipeline_name: str, dataset_id: UUID) -> bool:
    status = (getattr(item, "pipeline_status", None) or {}).get(pipeline_name) or {}
    return is_data_item_completed(status.get(str(dataset_id)))


def stage_items(stage: MemoryStage, items: list, dataset_id: UUID, incremental: bool) -> list:
    """The items a preparation run still has work for.

    An item routed to an empty task list (a DLT manifest, a code file) has
    none. With incremental loading neither has an item the stage already
    stamped, nor one whose extraction already completed: its chunks are
    stored and embedded, and running the earlier stages over it again (every
    document of a dataset cognified before this release) would only re-read
    and re-embed the corpus. Without incremental loading every document
    routed to the stage is processed again, as ``cognify(incremental_loading=False)``
    reprocesses everything.
    """
    selected = []
    for item in items:
        if not _stage_tasks_for(stage, item):
            continue
        if incremental and (
            _is_stamped(item, COGNIFY_PIPELINE_NAME, dataset_id)
            or _is_stamped(item, stage.pipeline_name, dataset_id)
        ):
            continue
        selected.append(item)
    return selected


async def _cognify_events(
    cognify_run: AsyncIterator[PipelineRunInfo], reached: dict[UUID, str]
) -> AsyncIterator[PipelineRunInfo]:
    """The cognify run's remaining events, its milestones logged and stamped."""
    async for run_info in cognify_run:
        if isinstance(run_info, PipelineRunCompleted):
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
        yield run_info


async def _dataset_chain(
    stages: list[MemoryStage], cognify_params: dict[str, Any], dataset_params: dict[str, Any]
) -> AsyncIterator[PipelineRunInfo]:
    """One dataset's runs, each event marked with the run the chain began with.

    The cognify run starts first and pauses after its start event: its
    ``pipeline_runs`` row exists from the beginning (status pollers see the
    dataset in progress, not the previous run's outcome), its LLM connection
    check runs before anything is written (a broken key fails the call at
    once, also in background mode, where the started event is what the
    caller gets back), and the data it selected under the lock is the data
    the preparation runs work on. Then the caching and indexing runs process
    the items that still need them (``stage_items``), and the cognify run
    resumes.
    """
    run_params = {k: v for k, v in cognify_params.items() if k in _SHARED_RUN_PARAMS}
    run_params.update(dataset_params)
    incremental = bool(cognify_params.get("incremental_loading", False))
    pipeline_name = cognify_params.get("pipeline_name", COGNIFY_PIPELINE_NAME)
    # The best state the dataset reached before extraction, for the failure line.
    reached: dict[UUID, str] = {}
    first_run_id: UUID | None = None

    def mark(run_info: PipelineRunInfo) -> PipelineRunInfo:
        # A caller that was handed the first run's id (background mode) keeps
        # receiving the later runs' events under it.
        nonlocal first_run_id
        if first_run_id is None:
            first_run_id = run_info.pipeline_run_id
        elif run_info.pipeline_run_id != first_run_id:
            run_info.chain_run_id = first_run_id
        return run_info

    cognify_run = run_pipeline(**{**cognify_params, **dataset_params})
    async with aclosing(cognify_run):
        try:
            started = await anext(cognify_run)
        except StopAsyncIteration:
            return
        # Captured before the event is handed on: the background executor
        # empties the started event's payload once it receives it.
        items = list(started.payload or []) if isinstance(started, PipelineRunStarted) else []
        dataset_id = started.dataset_id
        if isinstance(started, PipelineRunStarted):
            logger.info(
                "Building memory for dataset %r: extraction run %s (%s) started; the caching "
                "and indexing runs go first so the data answers searches before extraction.",
                started.dataset_name,
                started.pipeline_run_id,
                pipeline_name,
            )
        yield mark(started)

        for stage in stages:
            selected = stage_items(stage, items, dataset_id, incremental)
            if not selected:
                logger.info(
                    "Skipping %s for dataset %r: no item needs it (already processed, or not a "
                    "document).",
                    stage.pipeline_name,
                    started.dataset_name,
                )
                continue
            # The chain's id reaches the run's tasks too, so its per-task
            # progress reaches the subscriber of the cognify run.
            stage_params = {
                **run_params,
                "data": selected,
                "extras": {"chain_run_id": started.pipeline_run_id},
            }
            async with aclosing(_run_stage(stage, stage_params, reached)) as stage_events:
                async for run_info in stage_events:
                    yield mark(run_info)

        logger.info(
            "Extracting entities and relationships for dataset %r (%s, run %s).",
            started.dataset_name,
            pipeline_name,
            started.pipeline_run_id,
        )
        async with aclosing(_cognify_events(cognify_run, reached)) as events:
            async for run_info in events:
                yield mark(run_info)


async def run_memory_pipelines(
    caching_tasks: list | Callable[[Any], list] | None = None,
    indexing_tasks: list | Callable[[Any], list] | None = None,
    **cognify_params: Any,
) -> AsyncIterator[PipelineRunInfo]:
    """Yield, per dataset, the cognify run's start, the caching and indexing runs' events,
    then the rest of the cognify run.

    ``cognify_params`` are ``run_pipeline``'s keyword arguments for the cognify
    run (``tasks``, ``pipeline_name="cognify_pipeline"``, hooks, ...).
    ``caching_tasks`` and ``indexing_tasks`` are the earlier runs' task lists or
    per-item resolvers; ``None`` skips that run (``document_caching=False``,
    ``early_chunk_indexing=False``), and with both skipped this is exactly one
    cognify run.

    A dataset's runs are one unit under its lock, held from the first run's
    start to the last run's end, as a single cognify run held it: a concurrent
    ``cognify()`` or ``add()`` on the dataset waits for the whole chain, so it
    can neither process this call's files between two of its runs nor add
    files a later run would pick up. The runs inside take the lock's
    re-entrant path. Datasets are processed one after the other.

    A preparation run's ``PipelineRunCompleted`` is yielded as
    :class:`PipelineRunStageCompleted`, so a consumer that stops at "completed"
    keeps following into the run that comes next.
    """
    stages = []
    if caching_tasks is not None:
        stages.append(MemoryStage(**{**CACHING_STAGE.__dict__, "tasks": caching_tasks}))
    if indexing_tasks is not None:
        stages.append(MemoryStage(**{**INDEXING_STAGE.__dict__, "tasks": indexing_tasks}))

    if not stages:
        async with aclosing(_cognify_events(run_pipeline(**cognify_params), {})) as events:
            async for run_info in events:
                yield run_info
        return

    # What run_pipeline does before it resolves its datasets (the runs repeat
    # it, cheap once done). The LLM check is the cognify run's own, when it
    # starts first in each dataset's chain.
    await setup_and_check_environment(
        cognify_params.get("vector_db_config"),
        cognify_params.get("graph_db_config"),
        skip_connection_test=cognify_params.get("skip_connection_test", False),
        needs_llm=False,
    )
    user, datasets = await resolve_authorized_user_datasets(
        cognify_params.get("datasets"), cognify_params.get("user")
    )

    for dataset in datasets:
        chain = _dataset_chain(stages, cognify_params, {"datasets": [dataset.id], "user": user})
        if dataset.id in held_datasets.get():
            # An ancestor operation already owns this dataset's lock.
            async with aclosing(chain):
                async for run_info in chain:
                    yield run_info
            continue
        async with (
            await get_dataset_lock(dataset.id),
            aclosing(drive_marking_held(dataset.id, chain)) as source,
        ):
            async for run_info in source:
                yield run_info
