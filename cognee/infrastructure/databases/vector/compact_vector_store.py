"""Vector store compaction at the end of a cognify run."""

from __future__ import annotations

import asyncio

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
    return (
        f"{len(per_collection)} collection(s), fragments {removed} -> {added}, "
        f"{versions} old version(s) removed, left for later runs: {pending_tasks} task(s) "
        f"and {pending_versions} version(s)"
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


async def compact_vector_store() -> dict | None:
    """Compact the vector store bound to the current (dataset) context.

    Adapters that reclaim nothing on their own (LanceDB) implement ``compact``;
    for the rest it is ``VectorDBInterface``'s no-op. A failure raises; the
    cognify run it follows is already recorded as completed.

    Cancellation is not allowed to leave the pass running: in local mode the
    compaction runs in a thread and in subprocess mode inside the worker, and
    neither stops when this
    coroutine is cancelled. Leaving the dataset context right after would
    close the adapter (and its worker) under a rewrite that is still
    committing. So the pass is shielded and, on cancel, waited out -- through
    any further cancels -- before the cancellation propagates. That wait is
    short because a pass is bounded (``DEFAULT_MAX_TASKS_PER_RUN``,
    ``DEFAULT_MAX_VERSIONS_PER_RUN`` in ``cognee_db_workers.lancedb_compaction``).
    """
    vector_engine = await get_vector_engine_async()
    pass_task = asyncio.ensure_future(vector_engine.compact())
    try:
        stats = await asyncio.shield(pass_task)
    except asyncio.CancelledError:
        await _wait_out(pass_task)
        if not pass_task.cancelled() and pass_task.exception() is not None:
            logger.warning(
                "Vector store compaction failed during cancellation: %s", pass_task.exception()
            )
        raise
    if isinstance(stats, dict):
        (logger.info if _did_work(stats) else logger.debug)(
            "Vector store compaction: %s", _summarize(stats)
        )
    return stats
