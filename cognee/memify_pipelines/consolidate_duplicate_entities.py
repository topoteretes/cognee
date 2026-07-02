"""
Pipeline wrapper for the CONSOLIDATE memory task.

Usage
-----
    from cognee.memify_pipelines.consolidate_duplicate_entities import (
        consolidate_duplicate_entities_pipeline,
    )
    await consolidate_duplicate_entities_pipeline(
        similarity_threshold=0.92,
        dry_run=False,
    )
"""

import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.consolidate_duplicate_entities import consolidate_duplicate_entities


async def consolidate_duplicate_entities_pipeline(
    similarity_threshold: float = 0.92,
    protect_node_types: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    enrichment_tasks = [
        Task(
            consolidate_duplicate_entities,
            similarity_threshold=similarity_threshold,
            protect_node_types=protect_node_types or [],
            dry_run=dry_run,
        ),
    ]

    await cognee.memify(
        extraction_tasks=[],
        enrichment_tasks=enrichment_tasks,
        data=[{}],
    )