"""Orchestrate importing a MemorySource into Cognee.

Called by ``cognee.remember()`` when it receives a :class:`MemorySource`.
Heavy Cognee imports happen lazily inside the function to avoid import cycles
(remember -> migration -> remember).
"""

import time
from typing import TYPE_CHECKING, Optional

from cognee.modules.migration.loader import (
    store_imported_graph,
    translate_record_stream,
    wrap_graph_batch,
)
from cognee.modules.migration.sources.base import MemorySource
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.api.v1.remember.remember import RememberResult

logger = get_logger("migration.import")


async def import_memory_source(
    source: MemorySource,
    dataset_name: str = "main_dataset",
    user=None,
    run_in_background: bool = False,
    node_set: Optional[list] = None,
    **kwargs,
) -> "RememberResult":
    """Import all records from a memory source into a dataset.

    Returns a RememberResult summarizing what was imported. The import is
    idempotent: record-level deterministic ids (``data_id`` from
    external_system + external_id, node ids from entity names) make re-running
    an interrupted or repeated import safe.
    """
    from cognee.api.v1.remember.remember import RememberResult

    # Translate the record stream directly: no full raw-record list is kept,
    # so peak memory is bounded by the translation output alone.
    translation = await translate_record_stream(source.records(), source.mode)
    node_set = node_set or [f"import:{source.source_system}"]

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

    if translation.graph_batches:
        from cognee.modules.pipelines.tasks.task import Task
        from cognee.modules.run_custom_pipeline import run_custom_pipeline

        wrapped_batches = [
            wrap_graph_batch(batch, source.source_system, index)
            for index, batch in enumerate(translation.graph_batches)
        ]
        await run_custom_pipeline(
            tasks=[Task(store_imported_graph)],
            data=wrapped_batches,
            dataset=dataset_name,
            user=user,
            run_in_background=run_in_background,
            pipeline_name="migration_import_pipeline",
        )

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

    result = RememberResult(status="completed", dataset_name=dataset_name)
    result.items_processed = len(translation.data_items) + graph_nodes
    result.items.append(import_summary)
    result.elapsed_seconds = time.monotonic() - result._started_at
    return result
