"""Single-input pipeline entry point.

Adapter that feeds a single ``data`` value into the multi-item
worker-per-task executor in :mod:`worker_pipeline`. Multi-item callers
(the dataset-level :mod:`run_tasks`) should use ``run_worker_pipeline``
directly to feed the whole stream of data items into one shared set of
workers + queues.
"""

from typing import Any, AsyncGenerator, Optional

from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.users.models import User

from ..tasks.task import Task
from .worker_pipeline import _NO_DATA, _ErroredItem, run_worker_pipeline


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

    async for envelope in run_worker_pipeline(
        tasks=tasks,
        data_iterable=[head_payload],
        user=user,
        ctx=ctx,
        data_per_batch=1,
    ):
        if isinstance(envelope.value, _ErroredItem):
            raise envelope.value.exception
        yield envelope.value
