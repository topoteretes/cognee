"""decay_memory memify pipeline — exposes the time-decay/prune task as a memify enrichment pipeline.

Mirrors the whole-graph wrapper pattern (see consolidate_entity_descriptions / apply_feedback_weights):
a `Task`-wrapped task fn run via `cognee.memify(...)`. The decay task walks the whole graph itself,
so the extraction stage is an explicit no-op (otherwise memify falls back to its default
triplet-embedding extraction).
"""

from typing import List, Optional

import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.decay_memory import (
    decay_memory,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MIN_WEIGHT,
)

logger = get_logger("decay_memory_pipeline")


async def _passthrough(data):
    """No-op extraction stage: decay_memory reads the whole graph itself, so there's nothing to
    extract. Provided explicitly so memify() doesn't fall back to default triplet-embedding extraction."""
    return data


async def decay_memory_pipeline(
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    protect_node_types: Optional[List[str]] = None,
    dry_run: bool = True,
    dataset: str = "main_dataset",
):
    """Age node feedback weights by half-life and prune stale orphans, over a dataset's graph.

    dry_run=True (default) reports what *would* change without mutating the graph.
    """
    extraction_tasks = [Task(_passthrough)]
    enrichment_tasks = [
        Task(
            decay_memory,
            half_life_days=half_life_days,
            min_weight=min_weight,
            protect_node_types=protect_node_types,
            dry_run=dry_run,
        )
    ]
    result = await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # placeholder; decay_memory ignores it and walks the whole graph
        dataset=dataset,
    )
    logger.info(
        "decay_memory pipeline completed (dataset=%s, half_life_days=%s, dry_run=%s)",
        dataset,
        half_life_days,
        dry_run,
    )
    return result
