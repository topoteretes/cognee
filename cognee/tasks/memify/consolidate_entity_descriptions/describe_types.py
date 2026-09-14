import asyncio
from typing import Any

from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models.Entity import Entity
from cognee.shared.logging_utils import get_logger

from .apply_type_description import apply_type_description, group_entities_by_type
from .constants import (
    MAX_CONCURRENT_TYPE_LLM_CALLS,
    MAX_MEMBER_CARD_CHARS,
    MAX_MEMBERS_PER_TYPE_PROMPT,
    MAX_MERGE_PARTIAL_CHARS,
    MAX_NAMED_MEMBERS,
    MAX_PERSISTED_IS_A_CHARS,
    PARAGRAPH_MAX_COMPLETION_TOKENS,
    TOKENS_PER_IS_A_LINE,
    is_a_only_prompt_name,
    type_merge_prompt_name,
    type_prompt_name,
)
from .generate_type_description import generate_is_a_lines, generate_type_summary

logger = get_logger("consolidate_entity_descriptions")


async def generate_type_descriptions(
    entities: list[Entity],
    max_concurrent_calls: int = MAX_CONCURRENT_TYPE_LLM_CALLS,
    max_members_per_batch: int = MAX_MEMBERS_PER_TYPE_PROMPT,
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_member_card_chars: int = MAX_MEMBER_CARD_CHARS,
    max_merge_partial_chars: int = MAX_MERGE_PARTIAL_CHARS,
    max_persisted_is_a_chars: int = MAX_PERSISTED_IS_A_CHARS,
    max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS,
    tokens_per_is_a_line: int = TOKENS_PER_IS_A_LINE,
) -> list[Entity]:
    """Group rewritten entities by type, summarize each group, then point every
    member's is_a at the shared updated EntityType.

    Mutates and returns the same list of entities - that mutated list, not any
    return value, is what add_data_points consumes. Entities with no type pass
    through untouched.

    A type whose LLM calls fail is skipped rather than fatal: add_data_points
    runs after this task, so raising here would throw away every entity rewrite
    and every other type in the graph over one provider hiccup.
    """
    groups = group_entities_by_type(entities)
    system_prompt = render_prompt(type_prompt_name, {})
    merge_system_prompt = render_prompt(type_merge_prompt_name, {})
    is_a_system_prompt = render_prompt(is_a_only_prompt_name, {})
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def process_group(group: dict[str, Any]) -> None:
        entity_type = group["entity_type"]
        members = group["members"]
        description = await generate_type_summary(
            entity_type,
            members,
            system_prompt,
            merge_system_prompt,
            semaphore,
            max_members_per_batch,
            max_named_members,
            max_member_card_chars,
            max_merge_partial_chars,
            max_completion_tokens,
        )
        is_a_texts = await generate_is_a_lines(
            entity_type,
            members,
            description,
            is_a_system_prompt,
            semaphore,
            max_members_per_batch,
            max_member_card_chars,
            tokens_per_is_a_line,
        )
        apply_type_description(
            entity_type,
            members,
            description,
            is_a_texts,
            max_persisted_is_a_chars,
        )

    results = await asyncio.gather(
        *(process_group(group) for group in groups.values()), return_exceptions=True
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        logger.warning(
            "generate_type_descriptions: %d of %d types failed and were left unsummarized "
            "(first error: %r)",
            len(failures),
            len(results),
            failures[0],
        )

    return entities
