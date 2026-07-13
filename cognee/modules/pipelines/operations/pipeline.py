import asyncio
from uuid import UUID
from typing import AsyncIterator, Awaitable, Callable, Optional, Union

from cognee.infrastructure.locks import get_dataset_lock, held_datasets
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
from typing import Any

logger = get_logger("cognee.pipeline")

update_status_lock = asyncio.Lock()

# Per-dataset locks (shared with delete operations via cognee.infrastructure.locks)
# so concurrent runs on the SAME dataset are serialized: a run waits until any
# in-flight run for that dataset finishes, while different datasets still run in
# parallel. See cognee/infrastructure/locks/dataset_lock.py.


async def _drive_marking_held(dataset_id: UUID, source: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Yield from ``source`` while ``dataset_id`` is recorded as locked.

    A pipeline body runs its work in child tasks (``run_tasks`` -> ``create_task``),
    which copy the current context, so marking the dataset held *while the body
    advances* lets a nested run on the same dataset (e.g. ``cognify_session`` ->
    ``add()``/``cognify()``) see it as locked and take the re-entrant path. The
    marker is reset before every yield so it never leaks into the foreground driver
    across a yield — which in background mode would make a later run wrongly skip
    the lock. See ``held_datasets``.
    """
    marked = held_datasets.get() | {dataset_id}
    while True:
        token = held_datasets.set(marked)
        try:
            item = await source.__anext__()
        except StopAsyncIteration:
            return
        finally:
            held_datasets.reset(token)
        yield item


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
    data_cache: bool = False,
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
            data_cache=data_cache,
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
    data_cache=False,
):
    # The actual work of a single run, factored out so it can run either under
    # the per-dataset lock (normal case) or directly (re-entrant case below).
    async def _run_body():
        body_data = data if data else await get_dataset_data(dataset_id=dataset.id)

        if use_pipeline_cache:
            # Caching path: if this dataset's pipeline is already running or has
            # already completed, return that status instead of re-processing.
            # When caching is disabled the run always proceeds — concurrent runs
            # are kept safe by the per-dataset lock, not by this check.
            process_pipeline_status = await check_pipeline_run_qualification(
                dataset, body_data, pipeline_name
            )
            if process_pipeline_status:
                yield process_pipeline_status
                return

        pipeline_run = run_tasks(
            tasks,
            dataset.id,
            body_data,
            user,
            pipeline_name,
            incremental_loading=incremental_loading,
            data_per_batch=data_per_batch,
            rollback_handler=rollback_handler,
            llm_config=llm_config,
            embedding_config=embedding_config,
            data_cache=data_cache,
        )

        async for pipeline_run_info in pipeline_run:
            yield pipeline_run_info

    if dataset.id in held_datasets.get():
        # Re-entrant run: an ancestor pipeline run on this dataset already holds
        # the lock (e.g. cognify_session calls add()/cognify() on the same dataset
        # from inside a memify run). Re-acquiring the non-reentrant lock from the
        # same execution would self-deadlock, so run without re-locking — external
        # runs stay excluded by the lock the ancestor holds.
        async for run_info in _run_body():
            yield run_info
        return

    # External run: serialize on the per-dataset lock, marking the dataset held so
    # any nested run on it takes the re-entrant path above.
    async with await get_dataset_lock(dataset.id):
        async for run_info in _drive_marking_held(dataset.id, _run_body()):
            yield run_info
