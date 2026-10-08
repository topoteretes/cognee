from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.locks import get_dataset_lock, held_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.layers import validate_pipeline_tasks
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.setup_and_check_environment import (
    setup_and_check_environment,
)
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task, pipeline_needs_llm
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognee.pipeline")

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
    async with aclosing(source):
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
    tasks: list[Task] | Callable[[Any], list[Task]] | None = None,
    data=None,
    datasets: str | list[str] | list[UUID] | None = None,
    user: User | None = None,
    pipeline_name: str = "custom_pipeline",
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    rollback_handler: Callable[..., Awaitable[None]] | None = None,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    data_cache: bool = False,
    skip_connection_test: bool = False,
    needs_llm: bool = True,
    extras: dict | None = None,
    after_run_completed: Callable[[], Awaitable[Any]] | None = None,
    after_run: Callable[[], Awaitable[Any]] | None = None,
):
    """``tasks`` is either the task list every data item runs, or a callable
    mapping one item to its task list (a task resolver — see ``run_tasks``);
    items resolved to different lists still share one run per dataset.
    ``extras`` carries caller-resolved context into each item's PipelineContext.

    Whether the run needs the LLM drives the first-use LLM connection probe
    (skipped-but-never-marked-done when not needed; embeddings are always
    probed). For a task list it is derived from the tasks themselves — the
    union of ``Task.needs_llm`` — and the ``needs_llm`` parameter applies only
    when ``tasks`` is a resolver, whose caller must pass the union over every
    list the resolver can return.

    ``after_run_completed`` is awaited after each dataset's run completes, and
    ``after_run`` also when items in the run failed, both inside that dataset's
    database context (see ``run_tasks``)."""
    if tasks is None:
        raise ValueError(
            "run_pipeline requires tasks: a task list or a per-item task resolver callable"
        )
    if not callable(tasks):
        validate_pipeline_tasks(tasks)
        needs_llm = pipeline_needs_llm(tasks)
    await setup_and_check_environment(
        vector_db_config,
        graph_db_config,
        skip_connection_test=skip_connection_test,
        needs_llm=needs_llm,
    )

    user, authorized_datasets = await resolve_authorized_user_datasets(datasets, user)

    # TODO: If multiple datasets are provided, we currently run them sequentially to avoid overwhelming the system with too many concurrent pipeline runs.
    #       In the future, we could consider adding concurrency here with proper resource management and limits.
    for dataset in authorized_datasets:
        source = run_pipeline_per_dataset(
            dataset=dataset,
            user=user,
            tasks=tasks,
            data=data,
            pipeline_name=pipeline_name,
            incremental_loading=incremental_loading,
            data_per_batch=data_per_batch,
            rollback_handler=rollback_handler,
            llm_config=llm_config,
            embedding_config=embedding_config,
            data_cache=data_cache,
            extras=extras,
            after_run_completed=after_run_completed,
            after_run=after_run,
        )
        async with aclosing(source):
            async for run_info in source:
                yield run_info


async def run_pipeline_per_dataset(
    dataset: Dataset,
    user: User,
    tasks: list[Task] | Callable[[Any], list[Task]] | None = None,
    data: list[Data] | None = None,
    pipeline_name: str = "custom_pipeline",
    incremental_loading=False,
    data_per_batch: int = 20,
    rollback_handler: Callable[..., Awaitable[None]] | None = None,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    data_cache=False,
    extras: dict | None = None,
    after_run_completed: Callable[[], Awaitable[Any]] | None = None,
    after_run: Callable[[], Awaitable[Any]] | None = None,
):
    # The actual work of a single run, factored out so it can run either under
    # the per-dataset lock (normal case) or directly (re-entrant case below).
    async def _run_body():
        body_data = data if data else await get_dataset_data(dataset_id=dataset.id)

        # The run always proceeds. Concurrent runs on one dataset are serialized
        # by the per-dataset lock, and already-processed documents are skipped
        # per item by incremental loading, not by this dataset's run history.
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
            extras=extras,
            after_run_completed=after_run_completed,
            after_run=after_run,
        )

        async with aclosing(pipeline_run):
            async for pipeline_run_info in pipeline_run:
                yield pipeline_run_info

    async with aclosing(_run_body()) as body:
        if dataset.id in held_datasets.get():
            # An ancestor run already owns this dataset's non-reentrant lock.
            async for run_info in body:
                yield run_info
            return

        async with (
            await get_dataset_lock(dataset.id),
            aclosing(_drive_marking_held(dataset.id, body)) as source,
        ):
            async for run_info in source:
                yield run_info
