"""Once-per-pipeline-run vector store compaction. Best-effort by contract."""

from __future__ import annotations

import asyncio

from cognee.shared.logging_utils import get_logger

from .get_vector_engine import get_vector_engine

logger = get_logger("compact_vector_store")


def _summarize(stats: dict) -> str:
    if "skipped" in stats:
        return f"skipped ({stats['skipped']})"
    per_collection = [value for value in stats.values() if isinstance(value, dict)]
    removed = sum(int(value.get("fragments_removed", 0) or 0) for value in per_collection)
    added = sum(int(value.get("fragments_added", 0) or 0) for value in per_collection)
    pending = sum(
        max(0, int(value.get("planned_tasks", 0) or 0) - int(value.get("executed_tasks", 0) or 0))
        for value in per_collection
    )
    errors = sum(1 for value in per_collection if "error" in value)
    return (
        f"{len(per_collection)} collection(s), fragments {removed} -> {added}, "
        f"{pending} task(s) left for later runs, {errors} error(s)"
    )


async def compact_vector_store() -> dict | None:
    """Compact the vector store bound to the current (dataset) context.

    Adapters that reclaim nothing on their own (LanceDB) implement ``compact``;
    the rest have no such method and are skipped. A failure here must never
    fail the pipeline run that already succeeded: it is logged and swallowed.

    Cancellation is the one thing not swallowed, but it is not allowed to leave
    the pass running either: in local mode the compaction runs in a thread and
    in subprocess mode inside the worker, and neither stops when this
    coroutine is cancelled. The pipeline answers a cancel with a rollback that
    deletes rows from the same tables, and Lance rejects whichever of the two
    commits second. So the pass is shielded and, on cancel, awaited to
    completion (it is bounded work) before the cancellation propagates.
    """
    try:
        vector_engine = get_vector_engine()
        compact = getattr(vector_engine, "compact", None)
        if compact is None:
            return None
        pass_task = asyncio.ensure_future(compact())
    except Exception as exc:
        logger.warning("Vector store compaction skipped: %s", exc, exc_info=True)
        return None
    try:
        stats = await asyncio.shield(pass_task)
    except asyncio.CancelledError:
        if not pass_task.done():
            await asyncio.wait({pass_task})
        if pass_task.done() and not pass_task.cancelled() and pass_task.exception() is not None:
            logger.warning(
                "Vector store compaction failed during cancellation: %s", pass_task.exception()
            )
        raise
    except Exception as exc:
        logger.warning("Vector store compaction skipped: %s", exc, exc_info=True)
        return None
    if isinstance(stats, dict):
        summary = _summarize(stats)
        did_work = any(
            isinstance(value, dict) and int(value.get("executed_tasks", 0) or 0) > 0
            for value in stats.values()
        )
        (logger.info if did_work else logger.debug)("Vector store compaction: %s", summary)
    return stats
