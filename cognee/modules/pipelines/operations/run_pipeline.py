"""
run_pipeline — execute a list of BoundTasks as a pipeline.

This is the high-level API for the deferred-call pattern::

    from cognee.modules.pipelines.tasks.task import task
    from cognee.modules.pipelines.operations.run_pipeline import run_pipeline

    classify = task(classify_documents)
    extract  = task(extract_graph, batch_size=20)
    store    = task(add_data_points, batch_size=50)

    await run_pipeline([
        classify(),
        extract(graph_model=KnowledgeGraph),
        store(),
    ], data=raw_input)

Internally, it converts BoundTasks into Task objects and delegates to
the existing run_tasks_base machinery — preserving observability,
telemetry, provenance stamping, and error handling.
"""

import asyncio
from typing import Any, List, Optional

from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import BoundTask, Task
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base

logger = get_logger("run_pipeline")


async def run_pipeline(
    steps: List[BoundTask],
    *,
    data: Any = None,
    user: Optional[User] = None,
    dataset: Optional[str] = None,
    pipeline_name: str = "pipeline",
    context: Optional[dict] = None,
) -> list:
    """Execute a list of BoundTasks as a chained pipeline.

    Each step receives the output of the previous step as its first
    positional argument. Pre-bound kwargs from the BoundTask are merged
    into the Task's default_params.

    Args:
        steps: List of BoundTask objects (created by calling a TaskSpec).
        data: Initial input data for the first step.
        user: User context. Resolved to default user when None.
        dataset: Optional dataset name for context.
        pipeline_name: Name for logging/telemetry.
        context: Optional context dict passed to tasks.

    Returns:
        List of results from the final step.
    """
    if not steps:
        return [data] if data is not None else []

    # Validate all items are BoundTasks
    for i, step in enumerate(steps):
        if not isinstance(step, BoundTask):
            raise TypeError(
                f"Step {i} is {type(step).__name__}, expected BoundTask. "
                f"Did you forget to call the task? Use task_name() not task_name"
            )

    if user is None:
        user = await get_default_user()

    # Convert BoundTasks → Tasks with pre-bound kwargs
    tasks = []
    for bound in steps:
        inner = bound.task
        if bound.kwargs:
            # Merge BoundTask kwargs into the Task's default_params
            inner = inner.with_config(**bound.kwargs)
        tasks.append(inner)

    # Build typed context — pass caller-supplied context dict as extras
    ctx = PipelineContext(
        user=user,
        dataset=dataset,
        pipeline_name=pipeline_name,
        extras=context if isinstance(context, dict) else {},
    )

    # Delegate to existing run_tasks_base — gets us observability,
    # telemetry, provenance stamping, error handling for free.
    results = []
    async for result in run_tasks_base(tasks, data, user, ctx):
        results.append(result)

    return results
