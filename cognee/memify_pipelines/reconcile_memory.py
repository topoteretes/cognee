"""reconcile_memory memify pipeline — exposes the contradiction-resolution / supersede task as a memify
enrichment pipeline.

Mirrors the whole-graph wrapper pattern (see ``consolidate_entity_descriptions`` / ``apply_feedback_weights``):
a ``Task``-wrapped task fn run via ``cognee.memify(...)``. ``reconcile_memory`` walks the whole graph itself,
so the extraction stage is an explicit no-op (otherwise memify falls back to its default triplet-embedding
extraction).
"""

from typing import List, Optional

import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.reconcile_memory import (
    reconcile_memory,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEMOTE_FACTOR,
    DEFAULT_MAX_PAIRS,
    PREFER_RECENCY,
)

logger = get_logger("reconcile_memory_pipeline")


async def _passthrough(data):
    """No-op extraction stage: reconcile_memory reads the whole graph itself, so there's nothing to extract.
    Provided explicitly so memify() doesn't fall back to default triplet-embedding extraction."""
    return data


async def reconcile_memory_pipeline(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    prefer: str = PREFER_RECENCY,
    demote_factor: float = DEFAULT_DEMOTE_FACTOR,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    protect_node_types: Optional[List[str]] = None,
    dry_run: bool = True,
    dataset: str = "main_dataset",
):
    """Detect contradictory claims and supersede the stale ones, over a dataset's graph.

    dry_run=True (default) reports what *would* change without mutating the graph.
    """
    extraction_tasks = [Task(_passthrough)]
    enrichment_tasks = [
        Task(
            reconcile_memory,
            confidence_threshold=confidence_threshold,
            prefer=prefer,
            demote_factor=demote_factor,
            max_pairs=max_pairs,
            protect_node_types=protect_node_types,
            dry_run=dry_run,
        )
    ]
    result = await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # placeholder; reconcile_memory ignores it and walks the whole graph
        dataset=dataset,
    )
    logger.info(
        "reconcile_memory pipeline completed (dataset=%s, prefer=%s, dry_run=%s)",
        dataset,
        prefer,
        dry_run,
    )
    return result
