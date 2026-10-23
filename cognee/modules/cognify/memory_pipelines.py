"""Run the indexing pipeline, then the cognify pipeline (SDK-397).

``cognify()`` builds memory in two pipelines per dataset, each a run of its
own (its own ``pipeline_runs`` row and name, per-item completion stamps,
telemetry and rollback), so a dataset answers queries as soon as possible and
the answers improve once the graph exists:

1. ``indexing_pipeline``: classify, chunk, store and embed the chunks. No LLM
   call. When it completes the chunk searches answer (``CHUNKS``,
   ``CHUNKS_LEXICAL``, ``RAG_COMPLETION``, ``HYBRID_COMPLETION``).
2. ``cognify_pipeline``: extract entities and relationships, summarize, write
   the graph. It cuts the same chunks the indexing run stored, finds them in
   the vector store (``find_indexed_chunks``) and does not embed them twice.

Callers see one run per dataset, the cognify run, as before: it starts first
and pauses after its start event, so its ``pipeline_runs`` row exists from the
beginning (status pollers see the dataset in progress, not the previous run's
outcome), its LLM connection check runs before anything is written, and the
background executor hands back its run id. The indexing run then processes
the items that still need it (``indexing_items``) and the cognify run resumes.
The indexing run's own events stay inside this module: they are logged, and a
``PipelineRunProgress`` pushed to the cognify run's queue (as ``run_tasks``
pushes per-task progress) reports the indexing stage to its subscribers, so
nothing in the pipeline package knows about the chain.

The indexing run rolls back per item: an item that failed loses its chunks,
the items that completed keep theirs, and the cognify run stores the failed
item's chunks itself in its final write. A failed indexing run never decides
whether extraction runs, and a failed extraction leaves the indexed chunks
searchable (rollback selects artifacts by run id).

Each milestone is logged, so the memory's state can be read from the log
stream alone; ``cognee.modules.cognify.memory_status`` derives the same state
from the run rows.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from functools import partial
from typing import Any
from uuid import UUID

from cognee.infrastructure.locks import get_dataset_lock, held_datasets
from cognee.modules.cognify.rollback import cognify_rollback_handler
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
    PipelineRunProgress,
    PipelineRunStarted,
)
from cognee.modules.pipelines.operations.pipeline import drive_marking_held, run_pipeline
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME

logger = get_logger("cognify")

# run_pipeline parameters the indexing run shares with the cognify run. The
# task list, the pipeline name, the LLM need, the rollback policy, the extras
# and the after-run hooks are the cognify run's own.
_SHARED_RUN_PARAMS = (
    "vector_db_config",
    "graph_db_config",
    "incremental_loading",
    "data_per_batch",
    "llm_config",
    "embedding_config",
    "data_cache",
    "skip_connection_test",
)

# The stage name the cognify run's subscribers see while the indexing run works.
INDEXING_STAGE = "indexing_pipeline"

# Per-item rollback: a failed item never takes the other items' chunks with it.
_indexing_rollback_handler = partial(cognify_rollback_handler, keep_completed_data=True)


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


def _is_stamped(item: Any, pipeline_name: str, dataset_id: UUID) -> bool:
    status = (getattr(item, "pipeline_status", None) or {}).get(pipeline_name) or {}
    return is_data_item_completed(status.get(str(dataset_id)))


def indexing_items(
    indexing_tasks: list | Callable[[Any], list],
    items: list,
    dataset_id: UUID,
    incremental: bool,
) -> list:
    """The items the indexing run still has work for.

    An item routed to an empty task list (a DLT manifest, a code file) has
    none. With incremental loading neither has an item the indexing run
    already stamped, nor one whose extraction already completed: its chunks
    are stored and embedded, and indexing it again (every document of a
    dataset cognified before this release) would only re-read and re-embed the
    corpus. Without incremental loading every document routed to the run is
    processed again, as ``cognify(incremental_loading=False)`` reprocesses
    everything. Routing errors propagate: the cognify run would fail on the
    same item anyway, and it is better to fail before anything is written.
    """
    selected = []
    for item in items:
        tasks = indexing_tasks(item) if callable(indexing_tasks) else indexing_tasks
        if not tasks:
            continue
        if incremental and (
            _is_stamped(item, COGNIFY_PIPELINE_NAME, dataset_id)
            or _is_stamped(item, INDEXING_PIPELINE_NAME, dataset_id)
        ):
            continue
        selected.append(item)
    return selected


def _push_stage_progress(started: PipelineRunStarted, completed: int, total: int) -> None:
    """Tell the cognify run's subscriber that the indexing run works for it."""
    push_to_queue(
        started.pipeline_run_id,
        PipelineRunProgress(
            pipeline_run_id=started.pipeline_run_id,
            dataset_id=started.dataset_id,
            dataset_name=started.dataset_name,
            completed_items=completed,
            total_items=total,
            current_stage=INDEXING_STAGE,
        ),
    )


async def _run_indexing(
    indexing_tasks: list | Callable[[Any], list], run_params: dict[str, Any]
) -> bool:
    """Run the indexing pipeline to its end; True when it completed.

    Its events are logged, not yielded: the cognify run is the one callers
    follow. A failure is logged and swallowed, since extraction runs either
    way and its final write stores whatever the indexing run did not.
    """
    completed = False
    indexing_run = run_pipeline(
        tasks=indexing_tasks,
        pipeline_name=INDEXING_PIPELINE_NAME,
        needs_llm=False,
        rollback_handler=_indexing_rollback_handler,
        **run_params,
    )
    try:
        async with aclosing(indexing_run):
            async for run_info in indexing_run:
                if isinstance(run_info, PipelineRunStarted):
                    logger.info(
                        "Indexing dataset %r (%s, run %s): chunking and embedding %d item(s) "
                        "so chunk searches answer before entity extraction.",
                        run_info.dataset_name,
                        INDEXING_PIPELINE_NAME,
                        run_info.pipeline_run_id,
                        len(run_info.payload or []),
                    )
                elif isinstance(run_info, PipelineRunCompleted):
                    completed = True
                    processed, skipped = _item_counts(run_info)
                    logger.info(
                        "Dataset %r is indexed: %d item(s) chunked and embedded, %d already "
                        "were. Chunk searches (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, "
                        "HYBRID_COMPLETION) answer now; entity extraction is next, and "
                        "graph-based answers improve once it completes.",
                        run_info.dataset_name,
                        processed,
                        skipped,
                    )
                elif isinstance(run_info, PipelineRunErrored):
                    logger.warning(
                        "Indexing dataset %r failed (%s). The items that completed stay "
                        "searchable; entity extraction runs anyway and stores the other "
                        "items' chunks itself.",
                        run_info.dataset_name,
                        _failure(run_info),
                    )
    except Exception as error:
        # A run-level failure (setup, authorization, an exception escaping a
        # task). Extraction gets its own attempt and fails the same way if the
        # cause persists; that failure is the one callers see.
        logger.warning(
            "%s raised %s: %s. Continuing with entity extraction.",
            INDEXING_PIPELINE_NAME,
            type(error).__name__,
            error,
            exc_info=True,
        )
    return completed


def _log_cognify_event(run_info: PipelineRunInfo, indexed: bool) -> None:
    if isinstance(run_info, PipelineRunCompleted):
        processed, skipped = _item_counts(run_info)
        logger.info(
            "Dataset %r memory is fully built: entities and relationships extracted for "
            "%d item(s), %d already were. Chunk and graph searches answer.",
            run_info.dataset_name,
            processed,
            skipped,
        )
    elif isinstance(run_info, PipelineRunErrored):
        still = (
            "the indexed chunks stay searchable (CHUNKS, RAG_COMPLETION, HYBRID_COMPLETION)"
            if indexed
            else "nothing new is searchable"
        )
        logger.warning(
            "Entity extraction for dataset %r failed (%s); %s. Fix the cause and re-run "
            "cognify() to build the graph.",
            run_info.dataset_name,
            _failure(run_info),
            still,
        )


async def _dataset_chain(
    indexing_tasks: list | Callable[[Any], list],
    cognify_params: dict[str, Any],
    dataset_params: dict[str, Any],
) -> AsyncIterator[PipelineRunInfo]:
    """One dataset's cognify run, with the indexing run between its start and its work."""
    run_params = {k: v for k, v in cognify_params.items() if k in _SHARED_RUN_PARAMS}
    run_params.update(dataset_params)
    incremental = bool(cognify_params.get("incremental_loading", False))

    cognify_run = run_pipeline(**{**cognify_params, **dataset_params})
    async with aclosing(cognify_run):
        try:
            started = await anext(cognify_run)
        except StopAsyncIteration:
            return
        if not isinstance(started, PipelineRunStarted):
            # Nothing to prepare for (the run ended at once); pass it on as is.
            yield started
            async for run_info in cognify_run:
                yield run_info
            return

        # Captured before the event is handed on: the background executor
        # empties the started event's payload once it receives it.
        items = list(started.payload or [])
        selected = indexing_items(indexing_tasks, items, started.dataset_id, incremental)
        if selected:
            logger.info(
                "Building memory for dataset %r: extraction run %s (%s) started; the "
                "indexing run goes first so the data answers searches before extraction.",
                started.dataset_name,
                started.pipeline_run_id,
                COGNIFY_PIPELINE_NAME,
            )
        yield started

        indexed = False
        if selected:
            _push_stage_progress(started, 0, len(selected))
            indexed = await _run_indexing(indexing_tasks, {**run_params, "data": selected})
            _push_stage_progress(started, len(selected), len(selected))
        else:
            logger.info(
                "Skipping %s for dataset %r: no item needs it (already processed, or not a "
                "document).",
                INDEXING_PIPELINE_NAME,
                started.dataset_name,
            )

        logger.info(
            "Extracting entities and relationships for dataset %r (%s, run %s).",
            started.dataset_name,
            COGNIFY_PIPELINE_NAME,
            started.pipeline_run_id,
        )
        async for run_info in cognify_run:
            _log_cognify_event(run_info, indexed)
            yield run_info


async def run_memory_pipelines(
    indexing_tasks: list | Callable[[Any], list] | None = None,
    **cognify_params: Any,
) -> AsyncIterator[PipelineRunInfo]:
    """Yield, per dataset, the cognify run's events, with the indexing run between its start
    and its work.

    ``cognify_params`` are ``run_pipeline``'s keyword arguments for the cognify
    run (``tasks``, ``pipeline_name``, hooks, ...). ``indexing_tasks`` is the
    indexing run's task list or per-item resolver; ``None`` skips that run
    (``early_chunk_indexing=False``), and then this is exactly one cognify run.

    A dataset's two runs are one unit under its lock, held from the cognify
    run's start to its end, as a single cognify run held it: a concurrent
    ``cognify()`` or ``add()`` on the dataset waits for both, so it can neither
    process this call's files between the runs nor add files the cognify run
    would pick up. The runs inside take the lock's re-entrant path. Datasets
    are processed one after the other.
    """
    if indexing_tasks is None:
        cognify_run = run_pipeline(**cognify_params)
        async with aclosing(cognify_run):
            async for run_info in cognify_run:
                _log_cognify_event(run_info, False)
                yield run_info
        return

    # What run_pipeline does before it resolves its datasets (the runs repeat
    # it, cheap once done). The LLM check is the cognify run's own, which
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
        chain = _dataset_chain(
            indexing_tasks, cognify_params, {"datasets": [dataset.id], "user": user}
        )
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
