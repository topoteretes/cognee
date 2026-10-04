"""Vector store compaction at the end of a cognify run. Best-effort by contract."""

from __future__ import annotations

import asyncio

from cognee.infrastructure.background_tasks import register_background_task
from cognee.shared.logging_utils import get_logger

from .get_vector_engine import get_vector_engine_async

logger = get_logger("compact_vector_store")


def _summarize(stats: dict) -> str:
    if "skipped" in stats:
        return f"skipped ({stats['skipped']})"
    per_collection = [value for value in stats.values() if isinstance(value, dict)]
    removed = sum(int(value.get("fragments_removed", 0) or 0) for value in per_collection)
    added = sum(int(value.get("fragments_added", 0) or 0) for value in per_collection)
    pending_tasks = sum(
        max(0, int(value.get("planned_tasks", 0) or 0) - int(value.get("executed_tasks", 0) or 0))
        for value in per_collection
    )
    versions = sum(int(value.get("old_versions_removed", 0) or 0) for value in per_collection)
    pending_versions = sum(int(value.get("versions_pending", 0) or 0) for value in per_collection)
    errors = sum(1 for value in per_collection if "error" in value)
    return (
        f"{len(per_collection)} collection(s), fragments {removed} -> {added}, "
        f"{versions} old version(s) removed, left for later runs: {pending_tasks} task(s) "
        f"and {pending_versions} version(s), {errors} error(s)"
    )


def _did_work(stats: dict) -> bool:
    return any(
        isinstance(value, dict)
        and (
            int(value.get("executed_tasks", 0) or 0) > 0
            or int(value.get("old_versions_removed", 0) or 0) > 0
        )
        for value in stats.values()
    )


async def _wait_out(pass_task: asyncio.Future) -> None:
    """Wait for ``pass_task`` to finish, absorbing any further cancellations."""
    while not pass_task.done():
        try:
            await asyncio.wait({pass_task})
        except asyncio.CancelledError:
            continue


async def _compact_vector_store_now() -> dict | None:
    """Compact the vector store bound to the current (dataset) context.

    Adapters that reclaim nothing on their own (LanceDB) implement ``compact``;
    for the rest it is ``VectorDBInterface``'s no-op. A failure here must never
    fail the cognify run that already succeeded: it is logged and swallowed.

    Cancellation is the one thing not swallowed, but it is not allowed to leave
    the pass running either: in local mode the compaction runs in a thread and
    in subprocess mode inside the worker, and neither stops when this
    coroutine is cancelled. Leaving the dataset context right after would
    close the adapter (and its worker) under a rewrite that is still
    committing. So the pass is shielded and, on cancel, waited out -- through
    any further cancels -- before the cancellation propagates. That wait is
    short because a pass is bounded (``vector_db_compaction_max_tasks_per_run``,
    ``vector_db_compaction_max_versions_per_run``).
    """
    try:
        vector_engine = await get_vector_engine_async()
        pass_task = asyncio.ensure_future(vector_engine.compact())
    except Exception as exc:
        logger.warning("Vector store compaction skipped: %s", exc, exc_info=True)
        return None
    try:
        stats = await asyncio.shield(pass_task)
    except asyncio.CancelledError:
        await _wait_out(pass_task)
        if not pass_task.cancelled() and pass_task.exception() is not None:
            logger.warning(
                "Vector store compaction failed during cancellation: %s", pass_task.exception()
            )
        raise
    except Exception as exc:
        logger.warning("Vector store compaction skipped: %s", exc, exc_info=True)
        return None
    if isinstance(stats, dict):
        (logger.info if _did_work(stats) else logger.debug)(
            "Vector store compaction: %s", _summarize(stats)
        )
    return stats


async def compact_vector_store() -> asyncio.Task:
    """Start a compaction pass in the background and return its task without waiting.

    Called at the end of every cognify run, so neither the caller nor the dataset
    lock is held for the pass. The task copies the current context, so the pass
    compacts the dataset's own store. ``cognee.wait_for_background_tasks()`` and
    server shutdown wait for it, and so do the adapter's ``close`` and ``prune``.
    Dataset deletion does not: a dataset deleted while its pass runs can be left
    with stray files under its removed directory.
    """
    return register_background_task(asyncio.ensure_future(_compact_vector_store_now()))
