from collections.abc import Sequence
from typing import Any
from uuid import UUID

from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.engine.operations.setup import setup
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("memify")


async def run_memify_tasks(
    extraction_tasks: Sequence[Task],
    enrichment_tasks: Sequence[Task],
    data: Any | None = None,
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    user: User = None,
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    run_in_background: bool = False,
):
    """Run a fixed list of enrichment tasks over one dataset as ``memify_pipeline``.

    Internal runner for the pipelines in ``cognee/memify_pipelines`` that
    ``improve()`` stages call. It is not a public API: callers always pass
    their own tasks, and there are no default tasks. Custom tasks belong in
    ``run_custom_pipeline``.

    Args:
        extraction_tasks: Tasks that read the input the enrichment tasks work on.
        enrichment_tasks: Tasks that write the enrichment.
        data: Seed forwarded to the first extraction task. ``None`` becomes
            ``[{}]``, a placeholder for tasks that read what they need from the
            graph; any other value, an empty list included, is passed as given.
        dataset: Dataset name or UUID to process.
        user: User context for authentication and data access. Uses default if None.
        vector_db_config: Vector database override for the pipeline run.
        graph_db_config: Graph database override for the pipeline run.
        run_in_background: Return immediately and run the pipeline as a background task.

    Returns:
        The pipeline executor's result: a dict of dataset_id -> PipelineRunInfo
        in blocking mode, or the started runs in background mode.
    """
    tasks = [*extraction_tasks, *enrichment_tasks]
    if not tasks:
        raise ValueError("run_memify_tasks() needs at least one extraction or enrichment task")

    await setup()

    # The pipeline-run log writers INSERT the operation-record columns
    # (user_id, outcome, tokens, ...), so an existing database must be at the
    # current Alembic head before the first write — same gate as cognify().
    from cognee.modules.migrations.startup import run_migrations_and_block

    await run_migrations_and_block(dataset, user)

    user, authorized_dataset_list = await resolve_authorized_user_datasets(dataset, user)
    authorized_dataset = authorized_dataset_list[0]

    pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

    return await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=tasks,
        user=user,
        data=data if data is not None else [{}],
        datasets=[authorized_dataset.id],
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=False,
        data_cache=False,
        pipeline_name="memify_pipeline",
    )
