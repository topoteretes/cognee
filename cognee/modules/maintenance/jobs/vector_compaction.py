"""Vector store compaction after a cognify run (``LanceDBAdapter.compact``).

Every upsert leaves a small fragment and a table version behind, and LanceDB
reclaims neither on its own (issue #4684). The adapter merges small fragments
and deletes aged versions, bounded per pass by
``vector_db_compaction_max_tasks_per_run`` and
``vector_db_compaction_max_versions_per_run``; this job runs one pass and maps
its stats onto a ``JobResult``.
"""

from cognee.infrastructure.databases.vector import VectorDBInterface, get_vector_engine_async
from cognee.infrastructure.databases.vector.config import (
    get_vectordb_config,
    get_vectordb_context_config,
)
from cognee.infrastructure.databases.vector.create_vector_engine import (
    resolve_vector_adapter_class,
)

from ..job import BaseMaintenanceJob, MaintenanceContext
from ..result import REASON_BACKEND_UNSUPPORTED, REASON_DISABLED_BY_CONFIG, JobResult


class VectorCompactionJob(BaseMaintenanceJob):
    name = "vector_compaction"
    pipelines = frozenset({"cognify_pipeline"})
    # Compaction works on the whole vector store, which multi-user-off
    # deployments share across datasets.
    scope = "store"

    def gate(self, ctx: MaintenanceContext) -> str | None:
        # Configuration only: deciding to skip must not create the engine.
        if not get_vectordb_config().vector_db_compaction_enabled:
            return REASON_DISABLED_BY_CONFIG
        provider = str(get_vectordb_context_config().get("vector_db_provider") or "")
        if not adapter_compacts(provider):
            return REASON_BACKEND_UNSUPPORTED
        return None

    async def run(self, ctx: MaintenanceContext) -> JobResult:
        vector_engine = await get_vector_engine_async()
        stats = await vector_engine.compact()
        return result_from_stats(self.name, stats)


def adapter_compacts(provider: str) -> bool:
    """Whether the adapter class behind ``provider`` implements ``compact``.

    Decided from the class the vector factory would build
    (``resolve_vector_adapter_class``), never an instance: in-tree LanceDB, or
    a community adapter registered with ``use_vector_adapter`` that overrides
    ``VectorDBInterface.compact``. Adapters that inherit the no-op are skipped
    rather than called for nothing.
    """
    adapter_class = resolve_vector_adapter_class(provider)
    if not isinstance(adapter_class, type):
        return False
    return getattr(adapter_class, "compact", None) not in (None, VectorDBInterface.compact)


def result_from_stats(job: str, stats: dict | None) -> JobResult:
    """Map ``LanceDBAdapter.compact``'s per-collection stats onto a ``JobResult``.

    The adapter answers ``{"skipped": reason}`` when the whole pass is skipped
    (``remote_store``, ``in_progress``, ``lance_core_mismatch``, ...), otherwise
    ``{collection: stats}``: ``{"error": ...}`` for a collection that failed,
    and ``prune_error`` on one whose rewrite succeeded but whose prune failed.

    Every collection failed -> ``errored`` (with the counts). Some failed ->
    the status the work earns, with ``collection_errors`` / ``prune_errors``
    in the counts, which the runner logs at warning level.
    """
    if not stats:
        return JobResult.already_completed(job, collections=0)
    if "skipped" in stats:
        return JobResult.skipped(job, str(stats["skipped"]))
    per_collection = [value for value in stats.values() if isinstance(value, dict)]

    def total(key: str) -> int:
        return sum(int(value.get(key, 0) or 0) for value in per_collection)

    counts = {
        "collections": len(per_collection),
        "tasks_executed": total("executed_tasks"),
        "tasks_pending": sum(
            max(
                0,
                int(value.get("planned_tasks", 0) or 0) - int(value.get("executed_tasks", 0) or 0),
            )
            for value in per_collection
        ),
        "fragments_removed": total("fragments_removed"),
        "fragments_added": total("fragments_added"),
        "versions_removed": total("old_versions_removed"),
        "versions_pending": total("versions_pending"),
        "collection_errors": sum(1 for value in per_collection if "error" in value),
        "prune_errors": sum(1 for value in per_collection if "prune_error" in value),
    }
    if per_collection and counts["collection_errors"] == len(per_collection):
        return JobResult(
            job=job,
            status="errored",
            error=f"all {len(per_collection)} collection(s) failed",
            counts=counts,
        )
    did_work = counts["tasks_executed"] > 0 or counts["versions_removed"] > 0
    if did_work:
        return JobResult.completed(job, **counts)
    return JobResult.already_completed(job, **counts)
