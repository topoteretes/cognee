import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.modules.operations import scrub_error_message
from cognee.modules.operations.usage_accumulator import operation_usage_scope, parent_run_scope
from cognee.modules.pipelines.exceptions import AfterRunHookError, PipelineRunFailedError
from cognee.modules.pipelines.execution import (
    DataItemExecutor,
    ItemRun,
    RunSettings,
    get_data_item_executor,
    portable_item_check,
    portable_spec,
    run_items_with,
)
from cognee.modules.pipelines.layers.validate_pipeline_tasks import validate_pipeline_tasks
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
from cognee.modules.pipelines.operations import (
    log_pipeline_run_complete,
    log_pipeline_run_error,
    log_pipeline_run_progress,
    log_pipeline_run_start,
)
from cognee.modules.pipelines.run_ownership import pipeline_run_ownership
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry, telemetry_exception_properties, telemetry_guard
from cognee.tasks.ingestion import resolve_data_directories

from ..tasks.task import Task
from .run_tasks_data_item import _source_fields, run_tasks_data_item
from .run_tasks_with_telemetry import (
    PIPELINE_RUN_COMPLETED,
    PIPELINE_RUN_ERRORED,
    PIPELINE_RUN_STARTED,
    pipeline_run_telemetry_properties,
)

logger = get_logger("run_tasks(tasks: [Task], data)")

# Why a caller closed a run it holds paused (close_run_with_cause).
_close_cause: ContextVar[Exception | None] = ContextVar("pipeline_run_close_cause", default=None)


async def close_run_with_cause(run: AsyncGenerator, cause: Exception) -> None:
    """Close a paused run and record ``cause`` as its failure.

    Closing a run records it errored with ``GeneratorExit``; a caller that
    stops a run because of a known failure passes it here, so the run's row
    and its error class name the real cause.
    """
    token = _close_cause.set(cause)
    try:
        await run.aclose()
    finally:
        _close_cause.reset(token)


def writes_durable_before_push(*engines: Any) -> bool:
    """Whether a run's writes are durable as soon as they are written.

    An engine with ``push_to_s3`` (Ladybug, SQLite) may keep its writes in a
    local copy of an S3 database until the push after the item loop, so until
    then they exist in this process alone; its ``needs_push_to_s3`` says so,
    the same check its ``push_to_s3`` makes. Every such engine must define it:
    a missing one raises rather than passing writes off as durable.
    """
    return not any(engine.needs_push_to_s3() for engine in engines if hasattr(engine, "push_to_s3"))


def validated_once(resolve_tasks: Callable[[Any], list[Task]]) -> Callable[[Any], list[Task]]:
    """``resolve_tasks``, with each distinct task list it returns validated the first time."""
    validated_list_ids: set[int] = set()

    def resolve_validated(data_item: Any) -> list[Task]:
        item_tasks = resolve_tasks(data_item)
        if id(item_tasks) not in validated_list_ids:
            validate_pipeline_tasks(item_tasks)
            validated_list_ids.add(id(item_tasks))
        return item_tasks

    return resolve_validated


async def run_all_or_cancel(item_futures: list[asyncio.Task]) -> list[Any]:
    """The results of ``item_futures``. When one raises, or this is cancelled, the
    rest are cancelled and waited for: no item may still be writing once this raises."""
    try:
        return await asyncio.gather(*item_futures)
    except BaseException:
        for future in item_futures:
            future.cancel()
        await asyncio.gather(*item_futures, return_exceptions=True)
        raise


@dataclass
class DataItemRunner:
    """Runs data items of one pipeline run in this process.

    ``run_tasks`` runs its run's items through it, and ``run_pipeline_shard``
    the items of a run another process owns, so an item is run the same way
    wherever it runs.
    """

    settings: RunSettings
    dataset: Any
    user: User
    # Shared by the run's items: each records the stage it is at here.
    progress_state: dict[str, Any] | None = None
    # Awaited when an item ends, however it ended.
    on_item_done: Callable[[], Awaitable[None]] | None = None

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.settings.data_per_batch)

    async def run_item(self, data_item: Any, item_tasks: list[Task]) -> dict[str, Any] | None:
        run = self.settings
        try:
            async with self._semaphore:
                return await run_tasks_data_item(
                    data_item,
                    self.dataset,
                    item_tasks,
                    run.pipeline_name,
                    run.pipeline_id,
                    run.pipeline_run_id,
                    PipelineContext(
                        user=self.user,
                        data_item=data_item,
                        dataset=self.dataset,
                        pipeline_run_id=run.pipeline_run_id,
                        pipeline_name=run.pipeline_name,
                        # Copy per item: a shared dict would let one item's
                        # ctx.extras mutations leak into every other item.
                        extras=dict(run.extras),
                    ),
                    self.user,
                    run.incremental_loading,
                    run.data_cache,
                    self.progress_state,
                )
        finally:
            if self.on_item_done is not None:
                await self.on_item_done()

    def start(self, work_items: list[tuple[Any, list[Task]]]) -> list[asyncio.Task]:
        """Schedule ``(data_item, item_tasks)`` pairs, to await with ``run_all_or_cancel``."""
        return [
            asyncio.create_task(self.run_item(data_item, item_tasks))
            for data_item, item_tasks in work_items
        ]


async def run_tasks(
    tasks: list[Task] | Callable[[Any], list[Task]],
    dataset_id: UUID,
    data: list[Any] | None = None,
    user: User | None = None,
    pipeline_name: str = "unknown_pipeline",
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    extras: dict | None = None,
    rollback_handler: Callable[..., Awaitable[None]] | None = None,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    data_cache: bool = False,
    after_run_completed: Callable[[], Awaitable[Any]] | None = None,
    after_items_ran: Callable[[], Awaitable[Any]] | None = None,
    data_item_executor: DataItemExecutor | None = None,
):
    """Run a pipeline over a dataset as ONE logical run.

    ``tasks`` is either the task list every item runs, or a callable mapping
    one data item to its task list (a task resolver — like
    ``rollback_handler``, a caller-supplied policy that keeps the engine
    domain-blind). A constant list is just the degenerate resolver; items
    resolved to different lists still share this run's lifecycle — one run
    record, one database context, one rollback, one terminal status.

    ``after_run_completed`` is awaited once the run is recorded and reported
    complete, still inside this dataset's database context (cognify uses it
    to compact the vector store, see ``compact_vector_store``). It sits
    outside the run's error handling: it cannot fail, roll back, or re-mark a
    run that already completed. An exception it raises propagates to the
    caller as ``AfterRunHookError``; the run stays recorded as completed.

    ``after_items_ran`` is awaited before it, and also when items of the run failed,
    whether they reported or raised it. Cognify reconciles the document structure
    with it, which one failing item must not freeze. On a completed run its error
    propagates as ``AfterRunHookError`` too. On a failed run the run's own error wins:
    a raised one goes on with the hook's error as its cause, a reported one is returned
    and the hook's is logged.

    ``data_item_executor`` may run the run's data items in other processes when the
    run can leave this one (``execution.portable_spec`` says when); it comes before
    an executor registered for the process. The run stays this function's: its
    record, its database context, its rollback and its terminal status.
    """
    task_resolver = tasks if callable(tasks) else None
    if not user:
        user = await get_default_user()

    async with get_relational_engine().get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    with pipeline_run_ownership() as ownership:
        pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
        pipeline_run = await log_pipeline_run_start(
            pipeline_id, pipeline_name, dataset.id, user=user, recovery_token=ownership.token
        )
        pipeline_run_id = pipeline_run.pipeline_run_id
        # getattr (not attribute access) because unit tests stub
        # log_pipeline_run_start with plain namespaces lacking the field.
        run_started_at = getattr(pipeline_run, "started_at", None)

        telemetry_properties = pipeline_run_telemetry_properties(
            pipeline_id,
            pipeline_run_id,
            user.tenant_id,
            graph_extractor=(extras or {}).get("graph_extractor"),
            llm_config=llm_config,
            embedding_config=embedding_config,
        ) | {"pipeline_event_scope": "run"}
        with telemetry_guard():
            send_telemetry(PIPELINE_RUN_STARTED, user, additional_properties=telemetry_properties)

        run_usage = None
        database_ready = False
        run_finished = False
        items_failed = False
        # Whether this run's writes are durable: the rollback keeps the items the
        # run completed only then, since a kept item must still be there for the
        # next process. Set once the dataset's engines are known; becomes true
        # after the S3 push below where one is needed.
        writes_durable = False
        async with AsyncExitStack() as stack:
            try:
                yield PipelineRunStarted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    payload=data,
                )

                # Background execution resumes this generator in another task. Open
                # context scopes only after the handoff; the start event is already sent.
                run_usage = stack.enter_context(operation_usage_scope())
                stack.enter_context(parent_run_scope(pipeline_run_id))
                await stack.enter_async_context(
                    set_database_global_context_variables(
                        dataset.id,
                        dataset.owner_id,
                        llm_config=llm_config,
                        embedding_config=embedding_config,
                    )
                )
                database_ready = True
                writes_durable = writes_durable_before_push(
                    await get_graph_engine(), get_relational_engine()
                )
                if not isinstance(data, list):
                    data = [data]

                if data_cache or incremental_loading:
                    data = await resolve_data_directories(
                        data,
                        user=user,
                        dataset_id=dataset.id,
                        branch=(extras or {}).get("repo_branch"),
                    )

                # Build (item, item_tasks) work pairs: a resolver picks each
                # item's task list; a plain list applies uniformly. Validate each
                # DISTINCT resolved list once (the eager check in run_pipeline
                # covers only the plain-list case).
                resolve_tasks = (
                    validated_once(task_resolver) if task_resolver else lambda _item: tasks
                )
                work_items = [(item, resolve_tasks(item)) for item in data]

                # Item-level progress ("N of M files done"): a single counter shared
                # across all concurrently-gathered items, guarded by a lock since
                # several items finish interleaved under the semaphore.
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

                async def _record_item_progress(count: int = 1):
                    # An item that hard-raised (e.g. incremental loading with
                    # RAISE_INCREMENTAL_LOADING_ERRORS=true, the default) never
                    # reaches run_item's `return` — this is called from `finally`
                    # instead, so completed_items reaches total_items regardless of
                    # how the item ended. The error/success split itself still
                    # happens after gather() below; this only tracks "done", not
                    # outcome.
                    nonlocal completed_items
                    async with completed_items_lock:
                        previous_count = completed_items
                        completed_items += count
                        progress_count = completed_items

                    # A data item executor reports several items at once, so the
                    # throttle asks whether this step crossed a logging tick.
                    is_first_or_last = previous_count == 0 or progress_count == total_items
                    crossed_tick = (
                        previous_count // progress_log_every != progress_count // progress_log_every
                    )
                    if not (is_first_or_last or crossed_tick):
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
                    except Exception:
                        # Progress reporting must never fail the pipeline run.
                        logger.exception(
                            "Failed to log pipeline run progress",
                        )

                # Every item of the run goes through this runner, whoever starts it.
                runner = DataItemRunner(
                    RunSettings(
                        pipeline_name=pipeline_name,
                        pipeline_id=pipeline_id,
                        pipeline_run_id=pipeline_run_id,
                        incremental_loading=incremental_loading,
                        data_cache=data_cache,
                        data_per_batch=data_per_batch,
                        extras=extras if isinstance(extras, dict) else {},
                    ),
                    dataset,
                    user,
                    progress_state=progress_state,
                    on_item_done=_record_item_progress,
                )

                # A data item executor may run the items in other processes,
                # when the run can leave this one (see portable_spec): the one
                # this run was given, else the one registered for the process.
                executor = data_item_executor or get_data_item_executor()
                spec = None
                if executor is not None:
                    spec = portable_spec(
                        runner.settings,
                        task_resolver,
                        dataset.id,
                        user.id,
                        llm_config,
                        embedding_config,
                    )

                item_futures = []
                try:
                    if executor is None or spec is None:
                        item_futures = runner.start(work_items)
                        gathered = await run_all_or_cancel(item_futures)
                    else:
                        gathered = await run_items_with(
                            executor,
                            spec,
                            [item for item, _ in work_items],
                            ItemRun(
                                run_in_process=lambda items: run_all_or_cancel(
                                    runner.start([(item, resolve_tasks(item)) for item in items])
                                ),
                                record_progress=_record_item_progress,
                                portable_item=portable_item_check(task_resolver),
                                source_fields=lambda item: _source_fields(item, item),
                            ),
                        )
                except BaseException as gather_error:
                    # The items this process ran before the failure still report
                    # their result, so the rollback keeps what they completed. An
                    # executor's run reports none, the items it handed back
                    # included: the rollback takes a list as every result of the
                    # run, and that one would lack what other processes completed,
                    # so it would undo their work. With no list it goes by the
                    # completion stamps, which is right for that run.
                    if spec is None:
                        results = [
                            future.result()
                            for future in item_futures
                            if not future.cancelled()
                            and future.exception() is None
                            and future.result()
                        ]
                    # By default (RAISE_INCREMENTAL_LOADING_ERRORS) a failing item raises
                    # its own error here; a cancellation is not an item failure.
                    items_failed = isinstance(gather_error, Exception)
                    raise

                # Separate successes from unhandled exceptions
                results = []
                first_item_error: BaseException | None = None
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
                # data and emit two contradictory terminal events for one run. Until
                # the push succeeds the writes are not durable (``writes_durable``),
                # so a failure up to here rolls back the whole run, stamps included.
                graph_engine = await get_graph_engine()
                if hasattr(graph_engine, "push_to_s3"):
                    await graph_engine.push_to_s3()

                relational_engine = get_relational_engine()
                if hasattr(relational_engine, "push_to_s3"):
                    await relational_engine.push_to_s3()
                writes_durable = True

                await log_pipeline_run_complete(
                    pipeline_run_id,
                    pipeline_id,
                    pipeline_name,
                    dataset.id,
                    user=user,
                    started_at=run_started_at,
                    tokens_in=run_usage.tokens_in if run_usage else 0,
                    tokens_out=run_usage.tokens_out if run_usage else 0,
                    embedding_tokens=run_usage.embedding_tokens if run_usage else 0,
                )
                ownership.closed = True
                run_finished = True
                with telemetry_guard():
                    send_telemetry(
                        PIPELINE_RUN_COMPLETED, user, additional_properties=telemetry_properties
                    )

                yield PipelineRunCompleted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    data_ingestion_info=results,
                )

            except (Exception, asyncio.CancelledError, GeneratorExit) as error:
                if run_finished:
                    raise
                # Rollback needs a successfully opened database context. Failures
                # during setup and generator closure still receive a terminal record.
                if database_ready and callable(rollback_handler):
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
                            writes_durable=writes_durable,
                        )
                    except Exception:
                        logger.exception("Rollback errored")

                # Per-item failures arrive wrapped in a generic
                # PipelineRunFailedError; record and surface the ROOT cause so
                # the run record and the yielded run info say what actually
                # broke ("AuthenticationError: invalid api key"), not
                # "Pipeline run failed".
                root_error = getattr(error, "first_error", None) or error
                if isinstance(error, GeneratorExit):
                    root_error = _close_cause.get() or root_error
                with telemetry_guard():
                    send_telemetry(
                        PIPELINE_RUN_ERRORED,
                        user,
                        additional_properties=telemetry_properties
                        | telemetry_exception_properties(error),
                    )

                await log_pipeline_run_error(
                    pipeline_run_id,
                    pipeline_id,
                    pipeline_name,
                    dataset.id,
                    root_error,
                    user=user,
                    started_at=run_started_at,
                    tokens_in=run_usage.tokens_in if run_usage else 0,
                    tokens_out=run_usage.tokens_out if run_usage else 0,
                    embedding_tokens=run_usage.embedding_tokens if run_usage else 0,
                )

                ownership.closed = True
                if isinstance(error, GeneratorExit):
                    raise
                items_failed = items_failed or isinstance(error, PipelineRunFailedError)

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
                    if items_failed and after_items_ran is not None:
                        try:
                            await after_items_ran()
                        except Exception as hook_error:
                            # The run's own error is what callers map to a status, so it
                            # is the one raised; this log is the only trace of the hook's.
                            logger.exception("after_items_ran failed while the run was failing")
                            raise error from hook_error
                    raise

            # Reached only once the completed run's event has been consumed: a
            # close at that yield re-raises above (``run_finished``), and the
            # swallowed per-item failure path leaves ``run_finished`` unset.
            if run_finished:
                try:
                    if after_items_ran is not None:
                        await after_items_ran()
                    if after_run_completed is not None:
                        await after_run_completed()
                except Exception as hook_error:
                    raise AfterRunHookError(
                        [(dataset.id, dataset.name, hook_error)]
                    ) from hook_error
            elif items_failed and after_items_ran is not None:
                try:
                    await after_items_ran()
                except Exception:
                    # The reported failure was already yielded and is what callers act on
                    # (improve reads its budget error), so a broken hook must not replace it.
                    logger.exception("after_items_ran failed after a reported item failure")
