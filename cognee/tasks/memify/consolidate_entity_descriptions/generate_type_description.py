import asyncio

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity

from .constants import (
    MAX_MEMBER_CARD_CHARS,
    MAX_MEMBERS_PER_TYPE_PROMPT,
    MAX_MERGE_PARTIAL_CHARS,
    MAX_NAMED_MEMBERS,
    PARAGRAPH_MAX_COMPLETION_TOKENS,
    REASONING_HEADROOM_TOKENS,
    TOKENS_PER_IS_A_LINE,
    truncate,
)
from .models import EntityIsATexts, MemberIsAText, NodeDescription


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
    members: list[Entity],
    total_member_count: int,
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_member_card_chars: int = MAX_MEMBER_CARD_CHARS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Current description: {current_description or '(none)'}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {truncate(member.description, max_member_card_chars)}")
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


def batch_members(members: list[Entity], batch_size: int) -> list[list[Entity]]:
    return [members[i : i + batch_size] for i in range(0, len(members), batch_size)]


def build_type_merge_prompt(
    entity_type_name: str,
    total_member_count: int,
    partial_descriptions: list[str],
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_merge_partial_chars: int = MAX_MERGE_PARTIAL_CHARS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        (
            "You are given partial summaries, each covering a different subset of the "
            "members. Synthesize them into a single final summary following the same rules."
        ),
        "Partial summaries:",
    ]
    for index, partial in enumerate(partial_descriptions, start=1):
        lines.append(f"{index}. {truncate(partial, max_merge_partial_chars)}")
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
    members: list[Entity],
    total_member_count: int,
    max_member_card_chars: int = MAX_MEMBER_CARD_CHARS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Final type summary: {final_type_description}",
        f"Total member count: {total_member_count}",
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {truncate(member.description, max_member_card_chars)}")
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
        max_completion_tokens=member_count * tokens_per_is_a_line + REASONING_HEADROOM_TOKENS,
    )


async def generate_type_summary(
    entity_type: EntityType,
    members: list[Entity],
    system_prompt: str,
    merge_system_prompt: str,
    semaphore: asyncio.Semaphore,
    max_members_per_batch: int = MAX_MEMBERS_PER_TYPE_PROMPT,
    max_named_members: int = MAX_NAMED_MEMBERS,
    max_member_card_chars: int = MAX_MEMBER_CARD_CHARS,
    max_merge_partial_chars: int = MAX_MERGE_PARTIAL_CHARS,
    max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS,
) -> str:
    """Summarize a type's members, batching and merging when there are too many
    for a single prompt.

    Callers always pass the type's full member list - batching is an internal
    detail, not something the caller decides.

    ``semaphore`` bounds every individual LLM call, not just how many types are
    processed at once: a type with many batches would otherwise fire all of
    them in one unbounded asyncio.gather regardless of how many types are
    running concurrently.
    """
    total_member_count = len(members)
    batches = batch_members(members, max_members_per_batch)

    async def limited_type_call(batch: list[Entity]):
        async with semaphore:
            return await query_type_LLM(
                build_entity_type_prompt(
                    entity_type.name,
                    entity_type.description,
                    batch,
                    total_member_count,
                    max_named_members,
                    max_member_card_chars,
                ),
                system_prompt,
                max_completion_tokens,
            )

    partial_results = await asyncio.gather(*(limited_type_call(batch) for batch in batches))
    if len(batches) == 1:
        return partial_results[0].description

    merge_text = build_type_merge_prompt(
        entity_type.name,
        total_member_count,
        [result.description for result in partial_results],
        max_named_members,
        max_merge_partial_chars,
    )
    async with semaphore:
        merged = await query_type_merge_LLM(merge_text, merge_system_prompt, max_completion_tokens)
    return merged.description


async def generate_is_a_lines(
    entity_type: EntityType,
    members: list[Entity],
    final_description: str,
    is_a_system_prompt: str,
    semaphore: asyncio.Semaphore,
    max_members_per_batch: int = MAX_MEMBERS_PER_TYPE_PROMPT,
    max_member_card_chars: int = MAX_MEMBER_CARD_CHARS,
    tokens_per_is_a_line: int = TOKENS_PER_IS_A_LINE,
) -> list[MemberIsAText]:
    """One short is_a line per member, positioned against the finished summary.

    Always its own LLM call, never bundled with the summary. The two jobs need
    contradictory naming rules above max_named_members members: the summary
    must not name anyone, while every is_a line must start with a member's
    name - one response cannot honor both. And a per-batch partial summary
    only sees a fraction of the members, so a comparative claim ("handles the
    most packages") could be true for that batch and false once every member
    is considered. The lines therefore wait for the final, already-merged
    description.
    """
    total_member_count = len(members)
    batches = batch_members(members, max_members_per_batch)

    async def limited_is_a_call(batch: list[Entity]):
        async with semaphore:
            return await query_is_a_only_LLM(
                build_is_a_only_prompt(
                    entity_type.name,
                    final_description,
                    batch,
                    total_member_count,
                    max_member_card_chars,
                ),
                is_a_system_prompt,
                len(batch),
                tokens_per_is_a_line,
            )

    results = await asyncio.gather(*(limited_is_a_call(batch) for batch in batches))
    return [text for result in results for text in result.is_a_texts]
