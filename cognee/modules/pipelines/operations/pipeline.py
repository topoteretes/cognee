"""Dataset-aware pipeline orchestrator: run a task list over one or more datasets.

``run_pipeline`` is the runner behind ``add``, ``cognify``, ``improve`` and
``run_custom_pipeline``. A custom pipeline built from ``Task`` objects gets the same run
lifecycle as cognee's own pipelines, with no extra code:

* environment setup and the first-use LLM / embedding connection probe;
* user resolution (the default user when none is given) and dataset authorization, with
  write permission required on every dataset;
* per-dataset serialization through the dataset lock that delete operations also take,
  so runs and deletes on one dataset wait for each other while different datasets are
  not blocked (process-local, like the lock itself);
* per dataset, one logical run (``run_tasks``): a ``PipelineRun`` record, the dataset's
  own database context, up to ``data_per_batch`` items processed concurrently, an
  optional rollback on failure, and ``PipelineRunInfo`` events yielded to the caller.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.locks import get_dataset_lock, held_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods.get_selected_dataset_data import get_selected_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.exceptions import AfterRunHookError
from cognee.modules.pipelines.layers import validate_pipeline_tasks
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.setup_and_check_environment import (
    setup_and_check_environment,
)
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task, pipeline_needs_llm
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognee.pipeline")

# Per-dataset locks (shared with delete operations via cognee.infrastructure.locks)
# so concurrent runs on the SAME dataset are serialized: a run waits until any
# in-flight run for that dataset finishes, while different datasets still run in
# parallel. See cognee/infrastructure/locks/dataset_lock.py.


async def drive_marking_held(dataset_id: UUID, source: AsyncIterator[Any]) -> AsyncIterator[Any]:
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
    after_items_ran: Callable[[], Awaitable[Any]] | None = None,
):
    """Run ``tasks`` over each authorized dataset and yield its ``PipelineRunInfo`` events.

    An async generator: iterate it to drive the runs. Datasets run one after another,
    each under its dataset lock (see ``run_pipeline_per_dataset``), and each yields
    ``PipelineRunStarted`` followed by ``PipelineRunCompleted`` or ``PipelineRunErrored``.
    With ``incremental_loading`` or ``data_cache`` set, a data item already completed for
    ``pipeline_name`` in that dataset is skipped, recorded as
    ``PipelineRunAlreadyCompleted`` in the run's ``data_ingestion_info``.

    ``datasets`` takes names or ids. Omitted, the run covers every dataset the user may
    write to. A name the user has no dataset for creates that dataset; an unknown
    ``UUID``, or no dataset at all, raises ``DatasetNotFoundError``. ``data`` limits the
    run to those items; without it each dataset's run covers all of its rows.
    ``llm_config`` and ``embedding_config`` apply inside each dataset's database context.

    ``tasks`` is either the task list every data item runs, or a callable
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
    ``after_items_ran`` also when items in the run failed, both inside that dataset's
    database context (see ``run_tasks``). When either fails on a completed run, the
    remaining datasets still run and one ``AfterRunHookError`` naming every failed
    dataset is raised after them."""
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
    hook_failures = []
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
            after_items_ran=after_items_ran,
        )
        run_completed = False
        try:
            async with aclosing(source):
                async for run_info in source:
                    run_completed = isinstance(run_info, PipelineRunCompleted)
                    yield run_info
        except AfterRunHookError as error:
            # Only this dataset's own step after its completed run; one a task
            # raised (an improve bridge) failed the run and goes on as is.
            if not run_completed:
                raise
            # This dataset's run completed, so the next ones still run; every
            # failed step is raised together after them.
            hook_failures.extend(error.failures)
    if hook_failures:
        raise AfterRunHookError(hook_failures) from next(
            (error for *_, error in hook_failures if isinstance(error, BaseException)), None
        )


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
    after_items_ran: Callable[[], Awaitable[Any]] | None = None,
):
    """Run one dataset's pipeline under its dataset lock and yield its run events.

    The lock is the one delete operations take too, so a second run, add or delete on
    the same dataset waits until this run finishes, while other datasets proceed. It is
    process-local and does not serialize separate worker processes. Nested runs are
    supported: a task that starts another pipeline on the same dataset (e.g.
    ``cognify_session`` calling ``add()`` / ``cognify()``) reuses the lock its ancestor
    holds instead of deadlocking on it.

    ``Data`` rows in ``data`` are read again once the lock is held, so tasks see each row
    as it is now and rows deleted meanwhile are dropped; without ``data`` the run covers
    every row of the dataset. The remaining arguments pass through to ``run_tasks``.
    Callers normally use ``run_pipeline``, which authorizes the user and datasets first.
    """

    # The actual work of a single run, factored out so it can run either under
    # the per-dataset lock (normal case) or directly (re-entrant case below).
    async def _run_body():
        # Read under the dataset lock, so the run sees each selected row as it is now.
        body_data = (
            await get_selected_dataset_data(dataset.id, data)
            if data
            else await get_dataset_data(dataset_id=dataset.id)
        )

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
            after_items_ran=after_items_ran,
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
            aclosing(drive_marking_held(dataset.id, body)) as source,
        ):
            async for run_info in source:
                yield run_info
