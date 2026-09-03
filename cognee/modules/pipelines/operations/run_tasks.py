import asyncio

from typing import Any, Awaitable, Callable, List, Optional, Union
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.tasks.ingestion import resolve_data_directories
from cognee.modules.pipelines.layers.validate_pipeline_tasks import validate_pipeline_tasks
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
from cognee.modules.operations.usage_accumulator import operation_usage_scope, parent_run_scope
from cognee.modules.operations import scrub_error_message
from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
    log_pipeline_run_progress,
)
from .run_tasks_data_item import run_tasks_data_item
from ..tasks.task import Task


logger = get_logger("run_tasks(tasks: [Task], data)")


async def run_tasks(
    tasks: Union[List[Task], Callable[[Any], List[Task]]],
    dataset_id: UUID,
    data: Optional[List[Any]] = None,
    user: Optional[User] = None,
    pipeline_name: str = "unknown_pipeline",
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    extras: Optional[dict] = None,
    rollback_handler: Optional[Callable[..., Awaitable[None]]] = None,
    llm_config: Optional[LLMConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    data_cache: bool = False,
):
    """Run a pipeline over a dataset as ONE logical run.

    ``tasks`` is either the task list every item runs, or a callable mapping
    one data item to its task list (a task resolver — like
    ``rollback_handler``, a caller-supplied policy that keeps the engine
    domain-blind). A constant list is just the degenerate resolver; items
    resolved to different lists still share this run's lifecycle — one run
    record, one database context, one rollback, one terminal status.
    """
    task_resolver = tasks if callable(tasks) else None
    if not user:
        user = await get_default_user()

    async with get_relational_engine().get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
    pipeline_run = await log_pipeline_run_start(
        pipeline_id, pipeline_name, dataset.id, data, user=user
    )
    pipeline_run_id = pipeline_run.pipeline_run_id
    # getattr (not attribute access) because unit tests stub
    # log_pipeline_run_start with plain namespaces lacking the field.
    run_started_at = getattr(pipeline_run, "started_at", None)

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    # Note: Setting of global context has to be done after yielding PipelineRunStarted due to running in
    #       background mode requiring the pipeline run started yield.
    # parent_run_scope makes nested runs (a pipeline started by one of our
    # tasks, or a recorded operation called mid-pipeline) parent to THIS run,
    # mirroring how their tokens chain into run_usage.
    with operation_usage_scope() as run_usage, parent_run_scope(pipeline_run_id):
        async with set_database_global_context_variables(
            dataset.id,
            dataset.owner_id,
            llm_config=llm_config,
            embedding_config=embedding_config,
        ):
            try:
                if not isinstance(data, list):
                    data = [data]

                if data_cache or incremental_loading:
                    data = await resolve_data_directories(data, user=user, dataset_id=dataset.id)

                # Build (item, item_tasks) work pairs: a resolver picks each
                # item's task list; a plain list applies uniformly. Validate each
                # DISTINCT resolved list once (the eager check in run_pipeline
                # covers only the plain-list case).
                work_items = []
                validated_list_ids = set()
                for item in data:
                    item_tasks = task_resolver(item) if task_resolver else tasks
                    if task_resolver is not None and id(item_tasks) not in validated_list_ids:
                        validate_pipeline_tasks(item_tasks)
                        validated_list_ids.add(id(item_tasks))
                    work_items.append((item, item_tasks))

                # Semaphore-based concurrency: all items are scheduled at once,
                # but at most data_per_batch run concurrently at any time.
                semaphore = asyncio.Semaphore(data_per_batch)

                # Item-level progress ("N of M files done"): a single counter shared
                # across all concurrently-gathered items, guarded by a lock since
                # multiple _run_item calls finish interleaved under the semaphore.
                total_items = len(work_items)
                completed_items = 0
                completed_items_lock = asyncio.Lock()

                # Throttle the DB write, not just the in-memory count: one insert
                # per item would mean one extra write transaction per item, on top
                # of the pipeline's own writes — real contention on the default
                # SQLite relational backend, which serializes writers behind a
                # file lock. Capping at ~20 ticks per run keeps /status fresh
                # enough without adding write pressure proportional to batch size.
                # The first and last item always persist, so progress starts and
                # ends visible regardless of how coarse the throttle is.
                progress_log_every = max(1, total_items // 20)

                # Stage-level progress ("currently: graph extraction"): a shared
                # dict every item's stage ticks write current_stage into (see
                # _push_stage_progress in run_tasks_data_item.py). Items run
                # concurrently through the same task chain, so this is a
                # best-effort "what stage are things at" signal, not a per-item
                # value — good enough to answer /status without adding one DB
                # write per stage per item.
                progress_state = {"current_stage": None}

                async def _record_item_progress():
                    # An item that hard-raised (e.g. incremental loading with
                    # RAISE_INCREMENTAL_LOADING_ERRORS=true, the default) never
                    # reaches _run_item's `return` — this is called from `finally`
                    # instead, so completed_items reaches total_items regardless of
                    # how the item ended. The error/success split itself still
                    # happens after gather() below; this only tracks "done", not
                    # outcome.
                    nonlocal completed_items
                    async with completed_items_lock:
                        completed_items += 1
                        progress_count = completed_items

                    is_first_or_last = progress_count == 1 or progress_count == total_items
                    if not (is_first_or_last or progress_count % progress_log_every == 0):
                        return

                    try:
                        await log_pipeline_run_progress(
                            pipeline_run_id=pipeline_run_id,
                            pipeline_id=pipeline_id,
                            pipeline_name=pipeline_name,
                            dataset_id=dataset.id,
                            completed_items=progress_count,
                            total_items=total_items,
                            current_stage=progress_state["current_stage"],
                        )
                    except Exception as progress_error:
                        # Progress reporting must never fail the pipeline run.
                        logger.error(
                            f"Failed to log pipeline run progress: {progress_error}",
                            exc_info=True,
                        )

                async def _run_item(data_item, item_tasks):
                    try:
                        async with semaphore:
                            return await run_tasks_data_item(
                                data_item,
                                dataset,
                                item_tasks,
                                pipeline_name,
                                pipeline_id,
                                pipeline_run_id,
                                PipelineContext(
                                    user=user,
                                    data_item=data_item,
                                    dataset=dataset,
                                    pipeline_run_id=pipeline_run_id,
                                    pipeline_name=pipeline_name,
                                    # Copy per item: a shared dict would let one item's
                                    # ctx.extras mutations leak into every other item.
                                    extras=dict(extras) if isinstance(extras, dict) else {},
                                ),
                                user,
                                incremental_loading,
                                data_cache,
                                progress_state,
                            )
                    finally:
                        await _record_item_progress()

                gathered = await asyncio.gather(
                    *[
                        asyncio.create_task(_run_item(item, item_tasks))
                        for item, item_tasks in work_items
                    ],
                )

                # Separate successes from unhandled exceptions
                results = []
                first_item_error: Optional[BaseException] = None
                for i, result in enumerate(gathered):
                    if isinstance(result, BaseException):
                        logger.error(f"Item {i} failed: {result}", exc_info=result)
                        first_item_error = first_item_error or result
                        results.append(
                            {
                                "run_info": PipelineRunErrored(
                                    pipeline_run_id=pipeline_run_id,
                                    payload=repr(result),
                                    dataset_id=dataset.id,
                                    dataset_name=dataset.name,
                                    error_class=type(result).__name__,
                                    error_message=scrub_error_message(result),
                                ),
                            }
                        )
                    elif result:
                        results.append(result)

                # If any data item could not be processed propagate error
                errored_results = [
                    result
                    for result in results
                    if isinstance(result["run_info"], PipelineRunErrored)
                ]
                for errored_result in errored_results:
                    # The non-raising per-item path hands the root cause along
                    # in the result dict rather than as a gathered exception.
                    first_item_error = first_item_error or errored_result.get("error")
                if errored_results:
                    failure = PipelineRunFailedError(
                        message="Pipeline run failed. Data item could not be processed."
                    )
                    # Keep the underlying exception reachable: the outer handler
                    # logs/classifies the ROOT cause, not this generic wrapper —
                    # "Pipeline run failed" as the only recorded error text is
                    # exactly the observability gap this exists to close.
                    failure.first_error = first_item_error
                    raise failure

                # Flush durable storage BEFORE marking the run complete. If a push
                # fails it must be treated as a failure of this run (rollback +
                # PipelineRunErrored), not raised after the run has already been
                # reported as completed — which would both roll back already-completed
                # data and emit two contradictory terminal events for one run.
                graph_engine = await get_graph_engine()
                if hasattr(graph_engine, "push_to_s3"):
                    await graph_engine.push_to_s3()

                relational_engine = get_relational_engine()
                if hasattr(relational_engine, "push_to_s3"):
                    await relational_engine.push_to_s3()

                await log_pipeline_run_complete(
                    pipeline_run_id,
                    pipeline_id,
                    pipeline_name,
                    dataset.id,
                    data,
                    user=user,
                    started_at=run_started_at,
                    tokens_in=run_usage.tokens_in,
                    tokens_out=run_usage.tokens_out,
                )

                yield PipelineRunCompleted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    data_ingestion_info=results,
                )

            except (Exception, asyncio.CancelledError) as error:
                # asyncio.CancelledError is a BaseException (not an Exception)
                # since Python 3.8, so a bare `except Exception` misses it —
                # a cancelled run (deploy/restart, or a future disconnect-
                # triggered cancel) would otherwise never reach
                # log_pipeline_run_error below and stay stuck at
                # DATASET_PROCESSING_STARTED forever (CLO-365). Re-raised at
                # the end of this block either way, so cooperative
                # cancellation still propagates once cleanup is done.
                if callable(rollback_handler):
                    try:
                        await rollback_handler(
                            pipeline_run_id=pipeline_run_id,
                            pipeline_id=pipeline_id,
                            pipeline_name=pipeline_name,
                            dataset=dataset,
                            user=user,
                            data=data,
                            data_ingestion_info=locals().get("results"),
                            error=error,
                        )
                    except Exception as rollback_error:
                        logger.error("Rollback errored: %s", rollback_error, exc_info=True)

                # Per-item failures arrive wrapped in a generic
                # PipelineRunFailedError; record and surface the ROOT cause so
                # the run record and the yielded run info say what actually
                # broke ("AuthenticationError: invalid api key"), not
                # "Pipeline run failed".
                root_error = getattr(error, "first_error", None) or error

                await log_pipeline_run_error(
                    pipeline_run_id,
                    pipeline_id,
                    pipeline_name,
                    dataset.id,
                    data,
                    root_error,
                    user=user,
                    started_at=run_started_at,
                    tokens_in=run_usage.tokens_in,
                    tokens_out=run_usage.tokens_out,
                )

                yield PipelineRunErrored(
                    pipeline_run_id=pipeline_run_id,
                    payload=repr(root_error),
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    data_ingestion_info=locals().get(
                        "results"
                    ),  # Returns results if they exist or returns None
                    error_class=type(root_error).__name__,
                    error_message=scrub_error_message(root_error),
                )

                # In case of error during incremental loading of data just let the user know the pipeline Errored, don't raise error
                if not isinstance(error, PipelineRunFailedError):
                    raise error
