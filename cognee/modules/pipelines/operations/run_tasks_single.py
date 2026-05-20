"""Single-input pipeline entry point.

Adapter that feeds a single ``data`` value into the multi-item
worker-per-task executor in :mod:`worker_pipeline`. Multi-item callers
(the dataset-level :mod:`run_tasks`) should use ``run_worker_pipeline``
directly to feed the whole stream of data items into one shared set of
workers + queues.

Order guarantee: every stage is pinned to ``FixedWorkers(1)`` here so
that streamed yields from an upstream async-generator task arrive at
downstream stages in their yield order. Single-input pipelines gain no
throughput from per-stage parallelism (there is only one initial input),
and historical callers of ``run_tasks_single`` rely on ordered streaming
when chaining async generators.
"""

from typing import Any, AsyncGenerator, Optional

from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.users.models import User

from ..tasks.task import Task
from .worker_pipeline import (
    FixedWorkers,
    _NO_DATA,
    _ErroredItem,
    run_worker_pipeline,
)


def _pin_to_single_worker(task: Task) -> Task:
    """Return a clone of ``task`` with ``workers=FixedWorkers(1)`` unless the
    caller already set an explicit ``workers`` strategy. Preserves any other
    task_config / batch_size / timeout settings via ``with_config``."""
    if task.task_config.get("workers") is not None:
        return task
    return task.with_config(workers=FixedWorkers(1))


async def run_tasks_single(
    tasks: list[Task],
    data: Any = None,
    user: Optional[User] = None,
    ctx: Optional[PipelineContext] = None,
) -> AsyncGenerator[Any, None]:
    """Execute ``tasks`` on a single ``data`` value, streaming each yielded
    result from the final stage back to the caller.

    When ``data`` is ``None`` the first task is invoked with no positional
    arguments, matching today's behavior. Per-item errors surface as raised
    exceptions to preserve the existing contract.
    """
    if not tasks:
        yield data
        return

    head_payload = _NO_DATA if data is None else data
    serial_tasks = [_pin_to_single_worker(t) for t in tasks]

    async for envelope in run_worker_pipeline(
        tasks=serial_tasks,
        data_iterable=[head_payload],
        user=user,
        ctx=ctx,
        data_per_batch=1,
    ):
        if isinstance(envelope.value, _ErroredItem):
            raise envelope.value.exception
        yield envelope.value
