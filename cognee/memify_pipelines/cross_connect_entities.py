import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.cross_connect_entities import (
    cross_connect_entities,
    get_entity_nodes,
)


async def cross_connect_entities_pipeline(
    similarity_threshold: float = 0.5,
    overlap_threshold: float = 0.2,
    max_new_edges_per_node: int = 5,
    confidence_threshold: float = 0.7,
    dry_run: bool = False,
):
    extraction_tasks = [Task(get_entity_nodes)]

    enrichment_tasks = [
        Task(
            cross_connect_entities,
            similarity_threshold=similarity_threshold,
            overlap_threshold=overlap_threshold,
            max_new_edges_per_node=max_new_edges_per_node,
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
        )
    ]

    return await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],
    )
