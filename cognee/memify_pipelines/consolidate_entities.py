"""Memify pipeline that merges near-duplicate ``Entity`` nodes into one.

Mirrors the structure of the sibling graph-mutating enrichment pipelines (e.g.
``apply_feedback_weights``): it resolves the target dataset, enters that
dataset's database context, and runs a single detect (extraction) task followed
by a single merge (enrichment) task. The shared configuration is bound onto
both tasks so detection and merge agree on thresholds and on ``dry_run``.
"""

from typing import List, Optional

from cognee import memify
from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.consolidate_entities import (
    detect_entity_duplicates,
    merge_entity_duplicates,
)

logger = get_logger("consolidate_entities_pipeline")


async def consolidate_entities_pipeline(
    similarity_threshold: float = 0.85,
    dry_run: bool = False,
    protect_node_types: Optional[List[str]] = None,
    name_match: bool = True,
    top_k: int = 10,
    allow_cross_type: bool = False,
    user: Optional[User] = None,
    dataset: str = "main_dataset",
    run_in_background: bool = False,
):
    """Merge near-duplicate ``Entity`` nodes in an existing graph.

    Args:
        similarity_threshold: Minimum cosine similarity between two entity-name
            embeddings for them to be treated as the same entity.
        dry_run: When True, compute and log the merge plan but mutate nothing.
        protect_node_types: EntityType names that must never be merged.
        name_match: Also merge entities whose normalized names are identical.
        top_k: Max neighbors considered per entity during similarity clustering.
        allow_cross_type: When True, allow merging entities of different
            EntityTypes (off by default for safety).
        user: Acting user; the default user is used when omitted.
        dataset: Dataset name (or id) whose graph to consolidate.
        run_in_background: Forwarded to ``memify``.

    Returns:
        The ``memify`` pipeline result.
    """
    if not isinstance(similarity_threshold, (int, float)) or not 0 < similarity_threshold <= 1:
        raise CogneeValidationError(
            message="similarity_threshold must be in the range (0, 1]", log=False
        )

    # top_k drives the top-k neighbor optimization in clustering; a non-positive
    # or non-int value would silently fall back to an all-pairs scan, so reject
    # it up front. (bool is an int subclass — exclude it explicitly.)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise CogneeValidationError(message="top_k must be a positive integer", log=False)

    if protect_node_types is not None and (
        not isinstance(protect_node_types, list)
        or not all(isinstance(item, str) and item.strip() for item in protect_node_types)
    ):
        raise CogneeValidationError(
            message="protect_node_types must be a list of non-empty strings", log=False
        )

    if user is None:
        user = await get_default_user()

    datasets = await get_authorized_existing_datasets([dataset], "write", user)
    if not datasets:
        raise CogneeValidationError(
            message=f"User (id: {user.id}) has no write access to dataset: {dataset}",
            log=False,
        )
    target = datasets[0]

    config = {
        "similarity_threshold": similarity_threshold,
        "dry_run": dry_run,
        "protect_node_types": protect_node_types or [],
        "name_match": name_match,
        "top_k": top_k,
        "allow_cross_type": allow_cross_type,
    }

    async with set_database_global_context_variables(target.id, target.owner_id):
        extraction_tasks = [Task(detect_entity_duplicates, config=config)]
        enrichment_tasks = [Task(merge_entity_duplicates, config=config)]

        result = await memify(
            extraction_tasks=extraction_tasks,
            enrichment_tasks=enrichment_tasks,
            data=[{}],  # placeholder seed; the tasks read entities from the graph
            dataset=target.id,
            user=user,
            run_in_background=run_in_background,
        )

    logger.info(
        "consolidate_entities pipeline finished (dataset=%s, dry_run=%s, threshold=%.2f).",
        target.id,
        dry_run,
        similarity_threshold,
    )
    return result
