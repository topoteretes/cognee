"""Run the maintenance jobs a completed pipeline run triggers.

``run_maintenance`` is passed as a pipeline's ``after_run_completed`` hook (see
``run_tasks``), the same way a pipeline passes its ``rollback_handler``: the
pipeline engine never looks at the registry, and only pipelines that opt in
run maintenance (today: cognify). The hook runs after the run is recorded
complete, inside the dataset's database context and dataset lock, so a job
never overlaps a ``forget()``, a dataset delete or another run on the dataset.

Per job: the operator switch (``MAINTENANCE_JOBS_DISABLED``), then the job's
own gate, then the job, timed. Never raises -- a failed job is ``errored`` and
the next one still runs. Cancellation is the one thing passed on, but not by
abandoning work: a job's database work does not stop when its coroutine is
cancelled (a thread, a subprocess worker), and leaving the dataset context
right after would close the engine under it. So the running job is shielded
and waited out -- through any further cancels -- no later job starts, and only
then does the cancellation propagate. There is no timeout: every job bounds
its own work (see ``job.BaseMaintenanceJob``).
"""

import asyncio
import time
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from .config import get_maintenance_config
from .job import BaseMaintenanceJob, MaintenanceContext
from .result import (
    REASON_DISABLED_BY_CONFIG,
    REASON_GATE_ERRORED,
    REASON_SHARED_STORE,
    JobResult,
)

logger = get_logger("maintenance")


async def run_maintenance(
    *,
    pipeline_name: str,
    pipeline_run_id: UUID | None = None,
    dataset: Any = None,
    user: Any = None,
    last_in_invocation: bool = True,
    jobs: Sequence[BaseMaintenanceJob] | None = None,
    **_: Any,
) -> list[JobResult]:
    """Run, in registry order, the jobs triggered by a completed ``pipeline_name`` run.

    Accepts (and ignores) the rest of the keyword arguments ``run_tasks``
    passes its hook. ``last_in_invocation`` is False for every dataset but the
    last of a multi-dataset run (``run_pipeline`` sets it); store-scoped jobs
    use it to run once on a shared store. ``jobs`` replaces the registry, for
    tests.
    """
    if jobs is None:
        from .registry import DEFAULT_JOBS

        jobs = DEFAULT_JOBS
    triggered = [job for job in jobs if pipeline_name in job.pipelines]
    if not triggered:
        return []

    ctx = MaintenanceContext(
        pipeline_name=pipeline_name,
        pipeline_run_id=pipeline_run_id,
        dataset=dataset,
        user=user,
        last_in_invocation=last_in_invocation,
    )
    try:
        disabled = set(get_maintenance_config().jobs_disabled)
    except Exception:
        # A bad MAINTENANCE_* setting: when unsure, run nothing.
        logger.warning("maintenance: configuration could not be read", exc_info=True)
        return [JobResult.skipped(job.name, REASON_GATE_ERRORED) for job in triggered]

    results: list[JobResult] = []
    for job in triggered:
        if job.name in disabled:
            results.append(JobResult.skipped(job.name, REASON_DISABLED_BY_CONFIG))
            continue
        if job.scope == "store" and not last_in_invocation and _stores_are_shared():
            results.append(JobResult.skipped(job.name, REASON_SHARED_STORE))
            continue
        try:
            skip_reason = job.gate(ctx)
        except Exception:
            logger.warning("maintenance: gate of job '%s' failed", job.name, exc_info=True)
            skip_reason = REASON_GATE_ERRORED
        if skip_reason is not None:
            results.append(JobResult.skipped(job.name, skip_reason))
            continue

        job_task = asyncio.ensure_future(_run_job(job, ctx))
        try:
            results.append(await asyncio.shield(job_task))
        except asyncio.CancelledError:
            await _wait_out(job_task)
            logger.info(
                "maintenance after %s cancelled; job '%s' was let finish, later jobs not started",
                pipeline_name,
                job.name,
            )
            raise

    _log_summary(pipeline_name, results)
    return results


def _stores_are_shared() -> bool:
    """With multi-user off every dataset uses the one global set of databases."""
    from cognee.context_global_variables import backend_access_control_enabled

    return not backend_access_control_enabled()


async def _run_job(job: BaseMaintenanceJob, ctx: MaintenanceContext) -> JobResult:
    started_at = time.perf_counter()
    try:
        result = await job.run(ctx)
    except Exception as error:
        logger.warning("maintenance: job '%s' failed", job.name, exc_info=True)
        result = JobResult.errored(job.name, error)
    result.duration_ms = int((time.perf_counter() - started_at) * 1000)
    return result


async def _wait_out(task: asyncio.Future) -> None:
    """Wait for ``task`` to finish, absorbing any further cancellations."""
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            continue


def _log_summary(pipeline_name: str, results: list[JobResult]) -> None:
    """Warning when anything failed (even partly), info when work was done,
    debug when every job had nothing to do or was skipped."""
    if not results:
        return
    if any(result.has_failures for result in results):
        log = logger.warning
    elif any(result.status == "completed" for result in results):
        log = logger.info
    else:
        log = logger.debug
    log(
        "maintenance after %s: %s",
        pipeline_name,
        "; ".join(result.summary() for result in results),
    )
