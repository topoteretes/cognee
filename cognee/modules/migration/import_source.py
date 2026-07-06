"""Orchestrate importing a MemorySource into Cognee.

Called by ``cognee.remember()`` when it receives a :class:`MemorySource`.
Heavy Cognee imports happen lazily inside the function to avoid import cycles
(remember -> migration -> remember).

Two execution shapes:

- **streaming** (preserve mode + replayable source): records are passed over
  three times — data items chunk-stored via ``add()``, then a single pipeline
  task streams the graph in two passes (nodes, then facts) with bounded
  memory (see :func:`stream_graph_from_source`).
- **buffered** (re-derive/hybrid, or non-replayable sources): records are
  translated once into data items plus bounded graph batches, which run
  through the import pipeline together.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.migration.loader import (
    data_item_from_record,
    store_imported_graph,
    stream_graph_from_source,
    translate_record_stream,
    wrap_graph_batch,
)
from cognee.modules.migration.sources.base import MemorySource
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

if TYPE_CHECKING:
    from cognee.api.v1.remember.remember import RememberResult

logger = get_logger("migration.import")

# Data items are stored in chunks of this size in the streaming path so raw
# content never fully materializes in memory.
DATA_ITEMS_PER_ADD = 200

_GRAPH_RECORD_KINDS = ("entity", "fact", "raw_node")


def _pipeline_run_id(pipeline_result: Any) -> Optional[str]:
    """Extract the pipeline run id from a run_custom_pipeline return value.

    Blocking runs return ``{dataset_id: PipelineRunCompleted}``; background
    runs return ``{dataset_id: PipelineRunStarted}``. Both carry
    ``pipeline_run_id``.
    """
    if not pipeline_result:
        return None
    infos = pipeline_result.values() if isinstance(pipeline_result, dict) else [pipeline_result]
    for info in infos:
        run_id = getattr(info, "pipeline_run_id", None)
        if run_id:
            return str(run_id)
    return None


async def import_memory_source(
    source: MemorySource,
    dataset_name: str = "main_dataset",
    user=None,
    run_in_background: bool = False,
    node_set: Optional[list] = None,
    **kwargs,
) -> "RememberResult":
    """Import all records from a memory source into a dataset.

    Returns a RememberResult summarizing what was imported, carrying the
    migration pipeline's ``pipeline_run_id`` for polling when
    ``run_in_background=True`` (``status`` is then ``"started"`` and graph
    counts reflect scheduling, not completion). The import is idempotent:
    record-level deterministic ids (``data_id`` from external_system +
    external_id, node ids from entity names) make re-running an interrupted
    or repeated import safe.
    """
    from cognee.modules.migrations.startup import run_migrations_and_block

    # Imports are writes, so they take the same migration gate as
    # remember()/cognify() (the remember() MemorySource dispatch happens
    # before its own gate). This also records the data-migration revision —
    # stamping a fresh store at head — BEFORE the imported rows arrive;
    # without it the populated store has no recorded revision and the first
    # migration-aware startup replays the entire data chain over it.
    await run_migrations_and_block(dataset_name, user)

    node_set = node_set or [f"import:{source.source_system}"]

    if source.mode == "preserve" and getattr(source, "replayable", False):
        return await _import_streaming(source, dataset_name, user, run_in_background, node_set)
    return await _import_buffered(source, dataset_name, user, run_in_background, node_set, **kwargs)


async def _import_streaming(
    source: MemorySource,
    dataset_name: str,
    user,
    run_in_background: bool,
    node_set: list,
) -> "RememberResult":
    """Preserve-mode import with bounded memory.

    Pass A streams data items into chunked ``add()`` calls and counts every
    record kind; the graph then imports inside a single pipeline run whose
    task re-streams the source twice (nodes, then facts).
    """
    from cognee.api.v1.add import add
    from cognee.api.v1.remember.remember import RememberResult

    started_at = time.monotonic()

    counts: Dict[str, int] = {}
    pending: List[DataItem] = []
    data_items_stored = 0
    async for record in source.records():
        counts[record.kind] = counts.get(record.kind, 0) + 1
        data_item = data_item_from_record(record)
        if data_item is not None:
            pending.append(data_item)
            if len(pending) >= DATA_ITEMS_PER_ADD:
                await add(pending, dataset_name=dataset_name, user=user, node_set=node_set)
                data_items_stored += len(pending)
                pending = []
    if pending:
        await add(pending, dataset_name=dataset_name, user=user, node_set=node_set)
        data_items_stored += len(pending)

    logger.info("Importing from %s (mode=preserve, streaming): %s", source.source_system, counts)

    stats: Dict[str, int] = {"graph_nodes": 0, "graph_edges": 0, "skipped_facts": 0}
    pipeline_result = None
    has_graph_records = any(counts.get(kind) for kind in _GRAPH_RECORD_KINDS)
    if has_graph_records:
        from cognee.modules.pipelines.tasks.task import Task
        from cognee.modules.run_custom_pipeline import run_custom_pipeline

        async def stream_import_graph(items, ctx=None):
            return await stream_graph_from_source(source, stats, ctx=ctx)

        pipeline_data_item = DataItem(
            data={"source_system": source.source_system, "kind": "graph_stream"},
            label=f"migration-stream-{source.source_system}",
            external_metadata={
                "external_system": source.source_system,
                "kind": "graph_stream",
            },
            data_id=uuid5(NAMESPACE_OID, f"cogx-import:{source.source_system}:{dataset_name}"),
        )
        pipeline_result = await run_custom_pipeline(
            tasks=[Task(stream_import_graph)],
            data=[pipeline_data_item],
            dataset=dataset_name,
            user=user,
            run_in_background=run_in_background,
            pipeline_name="migration_import_pipeline",
        )

    backgrounded = run_in_background and has_graph_records
    if stats["skipped_facts"]:
        logger.warning(
            "Skipped %d facts with unresolvable UUID references during import from %s.",
            stats["skipped_facts"],
            source.source_system,
        )

    run_id = _pipeline_run_id(pipeline_result)
    import_summary = {
        "kind": "migration_import",
        "source_system": source.source_system,
        "mode": source.mode,
        "record_counts": counts,
        "graph_nodes": stats["graph_nodes"],
        "graph_edges": stats["graph_edges"],
        "skipped_facts": stats["skipped_facts"],
        "pipeline_run_id": run_id,
    }
    if backgrounded:
        # Graph counts reflect scheduling, not completion: the pipeline is
        # still running. Poll via pipeline_run_id.
        import_summary["graph_import"] = "running"

    result = RememberResult(
        status="started" if backgrounded else "completed", dataset_name=dataset_name
    )
    result.pipeline_run_id = run_id
    result.raw_result = pipeline_result
    result.items_processed = data_items_stored + stats["graph_nodes"]
    result.items.append(import_summary)
    result.elapsed_seconds = time.monotonic() - started_at
    return result


async def _import_buffered(
    source: MemorySource,
    dataset_name: str,
    user,
    run_in_background: bool,
    node_set: list,
    **kwargs,
) -> "RememberResult":
    """Translate the full record stream, then run data items and graph batches."""
    from cognee.api.v1.remember.remember import RememberResult

    started_at = time.monotonic()

    # Translate the record stream directly: no full raw-record list is kept,
    # so peak memory is bounded by the translation output alone.
    translation = await translate_record_stream(source.records(), source.mode)

    logger.info(
        "Importing %d records from %s (mode=%s): %s",
        sum(translation.counts.values()),
        source.source_system,
        source.mode,
        translation.counts,
    )
    if translation.skipped_facts:
        logger.warning(
            "Skipped %d facts with unresolvable UUID references during import from %s.",
            translation.skipped_facts,
            source.source_system,
        )

    graph_nodes = sum(len(batch["nodes"]) for batch in translation.graph_batches)
    graph_edges = sum(len(batch["edges"]) for batch in translation.graph_batches)
    import_summary = {
        "kind": "migration_import",
        "source_system": source.source_system,
        "mode": source.mode,
        "record_counts": translation.counts,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "skipped_facts": translation.skipped_facts,
    }

    pipeline_result = None
    if translation.graph_batches:
        from cognee.modules.pipelines.tasks.task import Task
        from cognee.modules.run_custom_pipeline import run_custom_pipeline

        wrapped_batches = [
            wrap_graph_batch(batch, source.source_system, index)
            for index, batch in enumerate(translation.graph_batches)
        ]
        pipeline_result = await run_custom_pipeline(
            tasks=[Task(store_imported_graph)],
            data=wrapped_batches,
            dataset=dataset_name,
            user=user,
            run_in_background=run_in_background,
            pipeline_name="migration_import_pipeline",
        )

    run_id = _pipeline_run_id(pipeline_result)
    import_summary["pipeline_run_id"] = run_id
    backgrounded = run_in_background and bool(translation.graph_batches)
    if backgrounded:
        import_summary["graph_import"] = "running"

    if translation.data_items and translation.cognify_data_items:
        from cognee.api.v1.remember.remember import remember

        result = await remember(
            translation.data_items,
            dataset_name,
            run_in_background=run_in_background,
            node_set=node_set,
            user=user,
            **kwargs,
        )
        result.items.append(import_summary)
        result.items_processed += graph_nodes
        # The nested remember() owns result.pipeline_run_id (the cognify run);
        # the migration pipeline's run id stays in the import summary.
        if result.pipeline_run_id is None:
            result.pipeline_run_id = run_id
        return result

    if translation.data_items:
        # Preserve mode: store raw content so it is available for a later
        # cognify, but do not run LLM extraction now.
        from cognee.api.v1.add import add

        await add(
            translation.data_items,
            dataset_name=dataset_name,
            user=user,
            node_set=node_set,
        )

    result = RememberResult(
        status="started" if backgrounded else "completed", dataset_name=dataset_name
    )
    result.pipeline_run_id = run_id
    result.raw_result = pipeline_result
    result.items_processed = len(translation.data_items) + graph_nodes
    result.items.append(import_summary)
    result.elapsed_seconds = time.monotonic() - started_at
    return result
