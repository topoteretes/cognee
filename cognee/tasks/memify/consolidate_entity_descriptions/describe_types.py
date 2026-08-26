import asyncio
from typing import Any, Dict, List

from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models.Entity import Entity

from .apply_type_description import apply_type_description, group_entities_by_type
from .constants import (
    MAX_CONCURRENT_TYPE_LLM_CALLS,
    MAX_MEMBERS_PER_TYPE_PROMPT,
    MAX_NAMED_MEMBERS,
    MAX_TYPE_TEXT_CHARS,
    PARAGRAPH_MAX_COMPLETION_TOKENS,
    TOKENS_PER_IS_A_LINE,
    is_a_only_prompt_name,
    type_merge_prompt_name,
    type_prompt_name,
)
from .generate_type_description import generate_type_description


async def generate_type_descriptions(
    entities: List[Entity],
    max_concurrent_calls: int = MAX_CONCURRENT_TYPE_LLM_CALLS,
    max_members_per_batch: int = MAX_MEMBERS_PER_TYPE_PROMPT,
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_type_text_chars: int = MAX_TYPE_TEXT_CHARS,
    max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS,
    tokens_per_is_a_line: int = TOKENS_PER_IS_A_LINE,
) -> List[Entity]:
    """Group rewritten entities by type, generate one type description per group,
    and point every member's is_a at the shared updated EntityType.

    Entities with no type pass through untouched. Returns the same list of
    entities (now with is_a updated where applicable), ready for add_data_points.
    """
    groups = group_entities_by_type(entities)
    system_prompt = render_prompt(type_prompt_name, {})
    merge_system_prompt = render_prompt(type_merge_prompt_name, {})
    is_a_system_prompt = render_prompt(is_a_only_prompt_name, {})
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def process_group(group: Dict[str, Any]) -> None:
        entity_type = group["entity_type"]
        members = group["members"]
        result = await generate_type_description(
            entity_type,
            members,
            system_prompt,
            merge_system_prompt,
            is_a_system_prompt,
            semaphore,
            max_members_per_batch,
            max_named_members,
            max_type_text_chars,
            max_completion_tokens,
            tokens_per_is_a_line,
        )
        apply_type_description(
            entity_type, members, result.description, result.is_a_texts, max_type_text_chars
        )

    await asyncio.gather(*(process_group(group) for group in groups.values()))

    return entities
