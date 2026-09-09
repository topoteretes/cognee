import asyncio
from typing import Any, AsyncIterable, AsyncGenerator, Callable, Dict, Union, Awaitable
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunErrored
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from cognee.modules.users.methods.get_default_user import get_default_user
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from cognee.shared.logging_utils import get_logger

AsyncGenLike = Union[
    AsyncIterable[Any],
    AsyncGenerator[Any, None],
    Callable[..., AsyncIterable[Any]],
    Callable[..., AsyncGenerator[Any, None]],
]

# Strong refs for fire-and-forget background pipeline tasks. The event loop only
# keeps weak references to tasks, so without anchoring here Python's gc can collect
# an in-flight task before it completes, silently aborting the background run. Tasks
# remove themselves on done, so this set's size tracks currently-running pipelines.
_BACKGROUND_PIPELINE_TASKS: set[asyncio.Task] = set()
logger = get_logger("pipeline_execution_mode")


def _handle_background_task_done(task: asyncio.Task) -> None:
    """Release a completed supervisor task and retrieve unexpected exceptions."""
    _BACKGROUND_PIPELINE_TASKS.discard(task)
    if task.cancelled():
        return

    error = task.exception()
    if error is not None:
        logger.error(
            "Background pipeline supervisor failed: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


async def _close_pipeline(pipeline: AsyncIterable[Any]) -> None:
    close = getattr(pipeline, "aclose", None)
    if not callable(close):
        return

    try:
        await close()
    except Exception:
        logger.exception("Failed to close a background pipeline generator")


async def _record_unhandled_pipeline_error(started_info, error: Exception) -> None:
    """Persist an error when a custom pipeline fails without doing so itself."""
    try:
        from cognee.modules.pipelines.methods import get_pipeline_run
        from cognee.modules.pipelines.operations import log_pipeline_run_error

        pipeline_run = await get_pipeline_run(started_info.pipeline_run_id)
        if pipeline_run is None:
            return

        run_info = pipeline_run.run_info if isinstance(pipeline_run.run_info, dict) else {}
        await log_pipeline_run_error(
            started_info.pipeline_run_id,
            pipeline_run.pipeline_id,
            pipeline_run.pipeline_name,
            started_info.dataset_id,
            run_info.get("data"),
            error,
        )
    except Exception:
        logger.exception(
            "Failed to persist background pipeline error for dataset %s",
            started_info.dataset_id,
        )


async def _drain_background_pipeline(pipeline, started_info) -> None:
    """Drain one pipeline without allowing its failure to abort later datasets."""
    terminal_emitted = False
    try:
        async for pipeline_run_info in pipeline:
            push_to_queue(pipeline_run_info.pipeline_run_id, pipeline_run_info)
            if isinstance(pipeline_run_info, (PipelineRunCompleted, PipelineRunErrored)):
                terminal_emitted = True
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.exception(
            "Background pipeline failed for dataset %s",
            started_info.dataset_id,
        )
        if not terminal_emitted:
            await _record_unhandled_pipeline_error(started_info, error)
            pipeline_run_info = PipelineRunErrored(
                pipeline_run_id=started_info.pipeline_run_id,
                dataset_id=started_info.dataset_id,
                dataset_name=started_info.dataset_name,
                payload=repr(error),
            )
            push_to_queue(pipeline_run_info.pipeline_run_id, pipeline_run_info)
    finally:
        await _close_pipeline(pipeline)


async def run_pipeline_blocking(pipeline: AsyncGenLike, **params) -> Dict[str, Any]:
    """
    Execute a pipeline synchronously (blocking until all results are consumed).

    This function iterates through the given pipeline (an async generator/iterable)
    until completion, aggregating the run information for each dataset.

    Args:
        pipeline (AsyncGenLike): The pipeline generator or callable producing async run information.
        **params: Arbitrary keyword arguments to be passed to the pipeline if it is callable.

    Returns:
        Dict[str, Any]:
            - If multiple datasets are processed, a mapping of dataset_id -> last run_info.
            - If no dataset_id is present in run_info, the run_info itself is returned.
    """
    agen = pipeline(**params) if callable(pipeline) else pipeline

    total_run_info: Dict[str, Any] = {}

    async for run_info in agen:
        dataset_id = getattr(run_info, "dataset_id", None)
        if dataset_id:
            total_run_info[dataset_id] = run_info
        else:
            total_run_info = run_info

    return total_run_info


async def run_pipeline_as_background_process(
    pipeline: AsyncGenLike,
    **params,
) -> Dict[str, Any]:
    """
    Execute one or more pipelines as background tasks.

    This function:
        1. Starts pipelines for each dataset (if multiple datasets are provided).
        2. Returns the initial "started" run information immediately.
        3. Continues executing the pipelines in the background,
           pushing run updates to a queue until each completes.

    Args:
        pipeline (AsyncGenLike): The pipeline generator or callable producing async run information.
        **params: Arbitrary keyword arguments to be passed to the pipeline if it is callable.
                  Expected to include "datasets", which may be a single dataset ID (str)
                  or a list of dataset IDs.

    Returns:
        Dict[str, Any]: A mapping of dataset_id -> initial run_info (with payload removed for serialization).
    """

    datasets = params.get("datasets", None)

    if not datasets:
        # If no datasets are provided, get all datasets user has write access to and run pipelines for all of them
        user = params.get("user", None)
        if user is None:
            user = await get_default_user()
        dataset_objects = await get_authorized_existing_datasets(None, "write", user)
        datasets = [dataset.id for dataset in dataset_objects]
    elif isinstance(datasets, str):
        datasets = [datasets]

    pipeline_run_started_info = {}

    async def handle_rest_of_the_run(pipeline_list):
        # Execute all provided pipelines one by one to avoid database write conflicts
        # TODO: Convert to async gather task instead of for loop when Queue mechanism for database is created
        try:
            for pipeline_run, started_info in pipeline_list:
                await _drain_background_pipeline(pipeline_run, started_info)
        finally:
            # Cancellation or an unexpected supervisor failure must also close
            # generators that have emitted STARTED but have not been drained yet.
            for pipeline_run, _ in pipeline_list:
                await _close_pipeline(pipeline_run)

    # Start all pipelines to get started status
    pipeline_list = []
    created_pipelines = []
    try:
        for dataset in datasets:
            call_params = dict(params)
            if "datasets" in call_params:
                call_params["datasets"] = dataset

            pipeline_run = pipeline(**call_params) if callable(pipeline) else pipeline
            created_pipelines.append(pipeline_run)

            # Save dataset Pipeline run started info
            run_info = await anext(pipeline_run)
            pipeline_run_started_info[run_info.dataset_id] = run_info

            if pipeline_run_started_info[run_info.dataset_id].payload:
                # Remove payload info to avoid serialization
                # TODO: Handle payload serialization
                pipeline_run_started_info[run_info.dataset_id].payload = []

            pipeline_list.append((pipeline_run, run_info))
    except BaseException:
        for created_pipeline in created_pipelines:
            await _close_pipeline(created_pipeline)
        raise

    # Send all started pipelines to execute one by one in background
    task = asyncio.create_task(handle_rest_of_the_run(pipeline_list=pipeline_list))
    _BACKGROUND_PIPELINE_TASKS.add(task)
    task.add_done_callback(_handle_background_task_done)

    return pipeline_run_started_info


def get_pipeline_executor(
    run_in_background: bool = False,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """
    Return the appropriate pipeline runner.

    Usage:
        run_fn = get_run_pipeline_fn(run_in_background=True)
        result = await run_fn(pipeline, **params)
    """
    return run_pipeline_as_background_process if run_in_background else run_pipeline_blocking
