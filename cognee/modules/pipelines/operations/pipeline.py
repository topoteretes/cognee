import asyncio
from uuid import UUID
from typing import Awaitable, Callable, Optional, Union

from cognee.modules.pipelines.layers.setup_and_check_environment import (
    setup_and_check_environment,
)

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.layers import validate_pipeline_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig

from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.check_pipeline_run_qualification import (
    check_pipeline_run_qualification,
)
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunStarted,
)
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.methods import reset_pipeline_run_status
from typing import Any

logger = get_logger("cognee.pipeline")

update_status_lock = asyncio.Lock()


async def run_pipeline(
    tasks: list[Task],
    data=None,
    datasets: Optional[Union[str, list[str], list[UUID]]] = None,
    user: Optional[User] = None,
    pipeline_name: str = "custom_pipeline",
    use_pipeline_cache: bool = False,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    rollback_handler: Optional[Callable[..., Awaitable[None]]] = None,
    llm_config: Optional[LLMConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
):
    validate_pipeline_tasks(tasks)
    await setup_and_check_environment(vector_db_config, graph_db_config)

    user, authorized_datasets = await resolve_authorized_user_datasets(datasets, user)

    # TODO: If multiple datasets are provided, we currently run them sequentially to avoid overwhelming the system with too many concurrent pipeline runs.
    #       In the future, we could consider adding concurrency here with proper resource management and limits.
    for dataset in authorized_datasets:
        async for run_info in run_pipeline_per_dataset(
            dataset=dataset,
            user=user,
            tasks=tasks,
            data=data,
            pipeline_name=pipeline_name,
            use_pipeline_cache=use_pipeline_cache,
            incremental_loading=incremental_loading,
            data_per_batch=data_per_batch,
            rollback_handler=rollback_handler,
            llm_config=llm_config,
            embedding_config=embedding_config,
        ):
            yield run_info


async def run_pipeline_per_dataset(
    dataset: Dataset,
    user: User,
    tasks: list[Task],
    data: Optional[list[Data]] = None,
    pipeline_name: str = "custom_pipeline",
    use_pipeline_cache=False,
    incremental_loading=False,
    data_per_batch: int = 20,
    rollback_handler: Optional[Callable[..., Awaitable[None]]] = None,
    llm_config: Optional[LLMConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
):
    if not data:
        data = await get_dataset_data(dataset_id=dataset.id)

    # Auto-recover datasets stuck in DATASET_PROCESSING_ERRORED state.
    #
    # When a previous cognify run fails (e.g. due to a transient DB error or
    # network issue), the pipeline_runs table retains a row with status
    # DATASET_PROCESSING_ERRORED. A subsequent cognify call on the same dataset
    # then raises RetryError[ProgrammingError] inside run_tasks because the
    # relational DB still holds stale state from the failed run.
    #
    # Calling reset_pipeline_run_status inserts a fresh DATASET_PROCESSING_INITIATED
    # row which clears the stale state and allows run_tasks to proceed normally.
    #
    # See: https://github.com/topoteretes/cognee/issues/3853
    task_status = await get_pipeline_status([dataset.id], pipeline_name)
    if task_status.get(str(dataset.id)) == PipelineRunStatus.DATASET_PROCESSING_ERRORED:
        logger.warning(
            "Dataset %s has a previous errored run for pipeline '%s'. "
            "Auto-resetting status to allow re-run. "
            "(See https://github.com/topoteretes/cognee/issues/3853)",
            dataset.id,
            pipeline_name,
        )
        await reset_pipeline_run_status(user.id, dataset.id, pipeline_name)

    process_pipeline_status = await check_pipeline_run_qualification(dataset, data, pipeline_name)
    if process_pipeline_status:
        # If pipeline was already processed or is currently being processed
        # return status information to async generator and finish execution
        if use_pipeline_cache:
            # If pipeline caching is enabled we do not proceed with re-processing
            yield process_pipeline_status
            return
        else:
            # If pipeline caching is disabled we always return pipeline started information and proceed with re-processing
            yield PipelineRunStarted(
                pipeline_run_id=process_pipeline_status.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=data,
            )

    pipeline_run = run_tasks(
        tasks,
        dataset.id,
        data,
        user,
        pipeline_name,
        incremental_loading=incremental_loading,
        data_per_batch=data_per_batch,
        rollback_handler=rollback_handler,
        llm_config=llm_config,
        embedding_config=embedding_config,
    )

    async for pipeline_run_info in pipeline_run:
        yield pipeline_run_info
