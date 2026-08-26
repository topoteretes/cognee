import asyncio
from typing import List

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity

from .constants import (
    MAX_MEMBERS_PER_TYPE_PROMPT,
    MAX_NAMED_MEMBERS,
    MAX_TYPE_TEXT_CHARS,
    PARAGRAPH_MAX_COMPLETION_TOKENS,
    TOKENS_PER_IS_A_LINE,
)
from .models import EntityIsATexts, EntityTypeDescription, NodeDescription


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_naming_instruction(total_member_count: int, max_named_members: int) -> str:
    """Decide, in code, whether members should be named - never ask the LLM to
    compare total_member_count against the threshold itself. An LLM asked to
    judge "is 5 at or below 5" is unreliable exactly at that boundary; a plain
    Python `<=` never is.
    """
    if total_member_count <= max_named_members:
        return (
            "You MUST name every member shown below individually, pairing each "
            "name with the fact that best distinguishes it from the others."
        )
    return "You MUST NOT name any individual member below - do not mention their names at all."


def build_entity_type_prompt(
    entity_type_name: str,
    current_description: str,
    members: List[Entity],
    total_member_count: int,
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_type_text_chars: int = MAX_TYPE_TEXT_CHARS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Current description: {current_description or '(none)'}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {_truncate(member.description, max_type_text_chars)}")
    return "\n".join(lines)


async def query_type_LLM(
    text_input, system_prompt, max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS
):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,
        response_model=NodeDescription,
        max_completion_tokens=max_completion_tokens,
    )


def batch_members(members: List[Entity], batch_size: int) -> List[List[Entity]]:
    return [members[i : i + batch_size] for i in range(0, len(members), batch_size)]


def build_type_merge_prompt(
    entity_type_name: str,
    total_member_count: int,
    partial_descriptions: List[str],
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_type_text_chars: int = MAX_TYPE_TEXT_CHARS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        "You are given partial summaries, each covering a different subset of the members. "
        "Synthesize them into a single final summary following the same rules.",
        "Partial summaries:",
    ]
    for index, partial in enumerate(partial_descriptions, start=1):
        lines.append(f"{index}. {_truncate(partial, max_type_text_chars)}")
    return "\n".join(lines)


async def query_type_merge_LLM(
    text_input, merge_system_prompt, max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS
):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=merge_system_prompt,
        response_model=NodeDescription,
        max_completion_tokens=max_completion_tokens,
    )


def build_is_a_only_prompt(
    entity_type_name: str,
    final_type_description: str,
    members: List[Entity],
    total_member_count: int,
    max_type_text_chars: int = MAX_TYPE_TEXT_CHARS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Final type summary: {final_type_description}",
        f"Total member count: {total_member_count}",
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {_truncate(member.description, max_type_text_chars)}")
    return "\n".join(lines)


async def query_is_a_only_LLM(
    text_input,
    is_a_system_prompt,
    member_count: int,
    tokens_per_is_a_line: int = TOKENS_PER_IS_A_LINE,
):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=is_a_system_prompt,
        response_model=EntityIsATexts,
        max_completion_tokens=member_count * tokens_per_is_a_line,
    )


async def generate_type_description(
    entity_type: EntityType,
    members: List[Entity],
    system_prompt: str,
    merge_system_prompt: str,
    is_a_system_prompt: str,
    semaphore: asyncio.Semaphore,
    max_members_per_batch: int = MAX_MEMBERS_PER_TYPE_PROMPT,
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_type_text_chars: int = MAX_TYPE_TEXT_CHARS,
    max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS,
    tokens_per_is_a_line: int = TOKENS_PER_IS_A_LINE,
) -> EntityTypeDescription:
    """Summarize a type's members, batching and merging the description when
    there are too many for a single prompt. Callers always pass the type's
    full member list here - batching is an internal detail, not something the
    caller decides.

    is_a lines are ALWAYS generated in their own call, separate from the
    description, never alongside it. The two jobs need contradictory naming
    rules whenever there are more than max_named_members members: the
    description must not name anyone, while every is_a line must start with a
    member's name - one response can't honor both at once. Beyond that
    correctness issue, a per-batch partial description also only sees a
    fraction of the members, so a comparative claim ("handles the most
    packages") could be true for that batch and false once every member is
    considered - another reason is_a lines must wait for the final,
    already-merged description before being generated.

    ``semaphore`` bounds every individual LLM call this function makes, not
    just how many types are processed at once - a type with many batches
    would otherwise fire all of them (and later all of its is_a calls) in one
    unbounded asyncio.gather, regardless of how many types are running
    concurrently.
    """
    total_member_count = len(members)
    batches = batch_members(members, max_members_per_batch)

    async def limited_type_call(batch: List[Entity]):
        async with semaphore:
            return await query_type_LLM(
                build_entity_type_prompt(
                    entity_type.name,
                    entity_type.description,
                    batch,
                    total_member_count,
                    max_named_members,
                    max_type_text_chars,
                ),
                system_prompt,
                max_completion_tokens,
            )

    if len(batches) == 1:
        async with semaphore:
            result = await query_type_LLM(
                build_entity_type_prompt(
                    entity_type.name,
                    entity_type.description,
                    members,
                    total_member_count,
                    max_named_members,
                    max_type_text_chars,
                ),
                system_prompt,
                max_completion_tokens,
            )
        final_description = result.description
    else:
        partial_results = await asyncio.gather(*(limited_type_call(batch) for batch in batches))
        partial_descriptions = [result.description for result in partial_results]
        merge_text = build_type_merge_prompt(
            entity_type.name,
            total_member_count,
            partial_descriptions,
            max_named_members,
            max_type_text_chars,
        )
        async with semaphore:
            merged = await query_type_merge_LLM(
                merge_text, merge_system_prompt, max_completion_tokens
            )
        final_description = merged.description

    async def limited_is_a_call(batch: List[Entity]):
        async with semaphore:
            return await query_is_a_only_LLM(
                build_is_a_only_prompt(
                    entity_type.name,
                    final_description,
                    batch,
                    total_member_count,
                    max_type_text_chars,
                ),
                is_a_system_prompt,
                len(batch),
                tokens_per_is_a_line,
            )

    is_a_results = await asyncio.gather(*(limited_is_a_call(batch) for batch in batches))
    is_a_texts = [text for result in is_a_results for text in result.is_a_texts]

    return EntityTypeDescription(description=final_description, is_a_texts=is_a_texts)
