from cognee import memify
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.tasks.memify import extract_subgraph_chunks, create_chunk_associations


logger = get_logger("chunk_associations_pipeline")


async def chunk_associations_pipeline(
    user: User = None,
    dataset: str = "main_dataset",
    similarity_threshold: float = 0.90,
    max_candidates_per_chunk: int = None,
    run_in_background: bool = False,
):
    """
    Pipeline for creating semantic associations between document chunks.

    This pipeline extracts chunks from the knowledge graph and creates weighted
    association edges between semantically similar chunks using LLM-based validation.

    Args:
        user: User context for authentication and data access (uses default if None)
        dataset: Dataset name to process (default: "main_dataset")
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider for association (default: 0.90)
        max_candidates_per_chunk: Maximum candidates per chunk. None = no limit, processes all above threshold (default: None)
        run_in_background: If True, runs asynchronously and returns immediately

    Returns:
        Pipeline execution result including pipeline_run_id for monitoring progress
    """
    extraction_tasks = [Task(extract_subgraph_chunks)]

    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=similarity_threshold,
            max_candidates_per_chunk=max_candidates_per_chunk,
            task_config={"batch_size": 1},
        ),
    ]

    # memify handles user authorization and context setup internally
    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset,
        user=user,
        run_in_background=run_in_background,
    )

    logger.info("Chunk associations pipeline completed")
    return result
