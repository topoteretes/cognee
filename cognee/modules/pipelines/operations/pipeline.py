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
from typing import Any

logger = get_logger("cognee.pipeline")

update_status_lock = asyncio.Lock()

# Per-dataset locks so concurrent pipeline runs on the SAME dataset are serialized:
# a run waits until any in-flight run for that dataset finishes, while different
# datasets still run in parallel.
# NOTE: process-local only (asyncio) — this does NOT protect against multiple
# processes/workers running against the same dataset. To be replaced by a
# cross-process mechanism (e.g. DB-backed lock) later.
_dataset_locks: dict[UUID, asyncio.Lock] = {}
_dataset_locks_guard = asyncio.Lock()


async def _get_dataset_lock(dataset_id: UUID) -> asyncio.Lock:
    """Return the asyncio.Lock for a dataset, creating it on first use."""
    async with _dataset_locks_guard:
        lock = _dataset_locks.get(dataset_id)
        if lock is None:
            lock = asyncio.Lock()
            _dataset_locks[dataset_id] = lock
        return lock


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
    # Serialize concurrent runs for the same dataset: hold the per-dataset lock
    # across the whole run so a second run for this dataset waits here until the
    # current one finishes.
    dataset_lock = await _get_dataset_lock(dataset.id)
    async with dataset_lock:
        if not data:
            data = await get_dataset_data(dataset_id=dataset.id)

        if use_pipeline_cache:
            # Caching path: if this dataset's pipeline is already running or has
            # already completed, return that status instead of re-processing.
            # When caching is disabled the run always proceeds — concurrent runs
            # are kept safe by the per-dataset lock above, not by this check.
            process_pipeline_status = await check_pipeline_run_qualification(
                dataset, data, pipeline_name
            )
            if process_pipeline_status:
                yield process_pipeline_status
                return

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
