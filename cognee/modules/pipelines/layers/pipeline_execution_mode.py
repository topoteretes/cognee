import asyncio
from typing import Any, AsyncIterable, AsyncGenerator, Callable, Dict, Union, Awaitable
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunErrored
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue

AsyncGenLike = Union[
    AsyncIterable[Any],
    AsyncGenerator[Any, None],
    Callable[..., AsyncIterable[Any]],
    Callable[..., AsyncGenerator[Any, None]],
]


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

    if isinstance(datasets, str):
        datasets = [datasets]

    pipeline_run_started_info = {}

    async def handle_rest_of_the_run(pipeline_list):
        # Execute all provided pipelines one by one to avoid database write conflicts
        # TODO: Convert to async gather task instead of for loop when Queue mechanism for database is created
        for pipeline in pipeline_list:
            while True:
                try:
                    pipeline_run_info = await anext(pipeline)

                    push_to_queue(pipeline_run_info.pipeline_run_id, pipeline_run_info)

                    if isinstance(pipeline_run_info, PipelineRunCompleted) or isinstance(
                        pipeline_run_info, PipelineRunErrored
                    ):
                        break
                except StopAsyncIteration:
                    break

    # Start all pipelines to get started status
    pipeline_list = []
    for dataset in datasets:
        call_params = dict(params)
        if "datasets" in call_params:
            call_params["datasets"] = dataset

        pipeline_run = pipeline(**call_params) if callable(pipeline) else pipeline

        # Save dataset Pipeline run started info
        run_info = await anext(pipeline_run)
        pipeline_run_started_info[run_info.dataset_id] = run_info

        if pipeline_run_started_info[run_info.dataset_id].payload:
            # Remove payload info to avoid serialization
            # TODO: Handle payload serialization
            pipeline_run_started_info[run_info.dataset_id].payload = []

        pipeline_list.append(pipeline_run)

    # Send all started pipelines to execute one by one in background
    asyncio.create_task(handle_rest_of_the_run(pipeline_list=pipeline_list))

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
