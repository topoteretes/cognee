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
from cognee.tasks.memify.consolidate_entity_descriptions.constants import (
    MAX_CONCURRENT_TYPE_LLM_CALLS,
    MAX_MEMBERS_PER_TYPE_PROMPT,
    MAX_NAMED_MEMBERS,
    MAX_TYPE_TEXT_CHARS,
    PARAGRAPH_MAX_COMPLETION_TOKENS as TYPE_PARAGRAPH_MAX_COMPLETION_TOKENS,
    TOKENS_PER_IS_A_LINE,
)
from cognee.tasks.memify.consolidate_entity_descriptions.rewrite_entities import (
    MAX_CONCURRENT_ENTITY_LLM_CALLS,
    MAX_NEIGHBOR_TEXT_CHARS,
    MAX_NEIGHBORS_IN_PROMPT,
    PARAGRAPH_MAX_COMPLETION_TOKENS as ENTITY_PARAGRAPH_MAX_COMPLETION_TOKENS,
)
from cognee.tasks.storage import add_data_points

logger = get_logger("consolidate_entity_descriptions_pipeline")


async def consolidate_entity_descriptions_pipeline(
    user: Optional[User] = None,
    dataset: str = DEFAULT_DATASET_NAME,
    run_in_background: bool = False,
    entity_max_concurrent_calls: int = MAX_CONCURRENT_ENTITY_LLM_CALLS,
    entity_max_neighbors: int = MAX_NEIGHBORS_IN_PROMPT,
    entity_max_neighbor_text_chars: int = MAX_NEIGHBOR_TEXT_CHARS,
    entity_description_max_completion_tokens: int = ENTITY_PARAGRAPH_MAX_COMPLETION_TOKENS,
    type_max_concurrent_calls: int = MAX_CONCURRENT_TYPE_LLM_CALLS,
    type_max_members_per_batch: int = MAX_MEMBERS_PER_TYPE_PROMPT,
    type_max_named_members: int = MAX_NAMED_MEMBERS,
    type_max_text_chars: int = MAX_TYPE_TEXT_CHARS,
    type_description_max_completion_tokens: int = TYPE_PARAGRAPH_MAX_COMPLETION_TOKENS,
    type_tokens_per_is_a_line: int = TOKENS_PER_IS_A_LINE,
):
    """Rewrite Entity descriptions from their graph neighborhood, then summarize
    each EntityType from its member Entities and write is_a edge text.

    Every size/budget cap below is a defensive backstop against pathological
    inputs (an unusually long description, an over-connected entity, a huge
    type), not a rigorously derived per-model token budget - cognee is
    model-agnostic, so there's no single number that's "correct" for every
    deployment. The defaults are reasonable starting points; override them
    per call site if your data or model needs something different, rather
    than editing the module constants.

    Args:
        user: Acting user; the default user is used when omitted.
        dataset: Dataset name (or id) whose graph to consolidate.
        run_in_background: Forwarded to ``memify``.
        entity_max_concurrent_calls: Max concurrent LLM calls while rewriting
            Entity descriptions (Phase 1).
        entity_max_neighbors: Max neighbors shown in one Entity's rewrite
            prompt.
        entity_max_neighbor_text_chars: Max characters of text per neighbor
            shown in that prompt.
        entity_description_max_completion_tokens: Output token budget for the
            Entity description call.
        type_max_concurrent_calls: Max concurrent LLM calls while summarizing
            EntityTypes (Phase 2/3) - bounds every individual call, not just
            how many types are processed at once.
        type_max_members_per_batch: Max members shown in one type-summary
            prompt before batching + merging kicks in.
        type_max_named_members: At or below this member count, the type
            summary names members individually; above it, it doesn't.
        type_max_text_chars: Max characters per member card, merge partial,
            and persisted is_a edge text.
        type_description_max_completion_tokens: Output token budget for the
            type description and merge calls.
        type_tokens_per_is_a_line: Output token budget per member for the
            is_a-only call - that call returns one line per member in the
            batch, not a single paragraph, so its total budget scales with
            batch size instead of being fixed.

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
            Task(
                generate_consolidated_entities,
                max_concurrent_calls=entity_max_concurrent_calls,
                max_neighbors=entity_max_neighbors,
                max_neighbor_text_chars=entity_max_neighbor_text_chars,
                max_completion_tokens=entity_description_max_completion_tokens,
            ),
            Task(
                generate_type_descriptions,
                max_concurrent_calls=type_max_concurrent_calls,
                max_members_per_batch=type_max_members_per_batch,
                max_named_members=type_max_named_members,
                max_type_text_chars=type_max_text_chars,
                max_completion_tokens=type_description_max_completion_tokens,
                tokens_per_is_a_line=type_tokens_per_is_a_line,
            ),
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
