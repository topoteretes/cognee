"""Memify pipeline that rewrites Entity descriptions and EntityType summaries.

Mirrors the structure of the sibling graph-mutating enrichment pipelines (e.g.
``consolidate_entities``): it resolves the target dataset, enters that
dataset's database context, and runs the extraction + enrichment tasks. The
actual work lives in ``cognee.tasks.memify.consolidate_entity_descriptions``.
"""

from typing import Optional

from cognee import memify
from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.consolidate_entity_descriptions import (
    generate_consolidated_entities,
    generate_type_descriptions,
    get_entities_with_neighborhood,
)
from cognee.tasks.storage import add_data_points

logger = get_logger("consolidate_entity_descriptions_pipeline")


async def consolidate_entity_descriptions_pipeline(
    user: Optional[User] = None,
    dataset: str = DEFAULT_DATASET_NAME,
    run_in_background: bool = False,
):
    """Rewrite Entity descriptions from their graph neighborhood, then summarize
    each EntityType from its member Entities and write is_a edge text.

    Args:
        user: Acting user; the default user is used when omitted.
        dataset: Dataset name (or id) whose graph to consolidate.
        run_in_background: Forwarded to ``memify``.

    Returns:
        The ``memify`` pipeline result.
    """
    if user is None:
        user = await get_default_user()

    datasets = await get_authorized_existing_datasets([dataset], "write", user)
    if not datasets:
        raise CogneeValidationError(
            message=f"User (id: {user.id}) has no write access to dataset: {dataset}",
            log=False,
        )
    target = datasets[0]

    async with set_database_global_context_variables(target.id, target.owner_id):
        extraction_tasks = [Task(get_entities_with_neighborhood)]

        enrichment_tasks = [
            Task(generate_consolidated_entities),
            Task(generate_type_descriptions),
            Task(add_data_points),
        ]

        result = await memify(
            extraction_tasks=extraction_tasks,
            enrichment_tasks=enrichment_tasks,
            data=[{}],  # A placeholder to prevent fetching the entire graph
            dataset=target.id,
            user=user,
            run_in_background=run_in_background,
        )

    logger.info("consolidate_entity_descriptions pipeline finished (dataset=%s).", target.id)
    return result
