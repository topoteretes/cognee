import asyncio
from typing import Any, Dict, List, Optional, Union

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity

from .models import EntityIsATexts, EntityTypeDescription, MemberIsAText

type_prompt_name = "consolidate_entity_type_details.txt"
type_merge_prompt_name = "consolidate_entity_type_merge.txt"
is_a_only_prompt_name = "consolidate_entity_is_a_only.txt"
MAX_CONCURRENT_TYPE_LLM_CALLS = 10
MAX_NAMED_MEMBERS = 5
MAX_MEMBERS_PER_TYPE_PROMPT = 50


def _entity_type_of(is_a: Optional[Union[EntityType, tuple]]) -> Optional[EntityType]:
    """Unwrap is_a to its EntityType, whether it's bare or (Edge, EntityType)."""
    if isinstance(is_a, tuple):
        return is_a[1]
    return is_a


def _is_a_relation_type(relation: Any) -> Optional[EntityType]:
    """Return the EntityType of an is_a-tagged (Edge, EntityType) tuple in relations, else None."""
    if (
        isinstance(relation, tuple)
        and len(relation) == 2
        and isinstance(relation[0], Edge)
        and relation[0].relationship_type == "is_a"
        and isinstance(relation[1], EntityType)
    ):
        return relation[1]
    return None


def all_entity_types(entity: Entity) -> List[EntityType]:
    """Every type this entity belongs to - the single one on is_a (the common
    case), or all of them from relations for an entity with more than one type
    (is_a is None in that case; see build_entity)."""
    primary = _entity_type_of(entity.is_a)
    if primary is not None:
        return [primary]
    return [
        entity_type
        for relation in entity.relations
        if (entity_type := _is_a_relation_type(relation)) is not None
    ]


def group_entities_by_type(entities: List[Entity]) -> Dict[str, Dict[str, Any]]:
    """Group rewritten entities by their EntityType id.

    An entity with multiple types is registered as a member of every one of
    its type groups, not just one - see all_entity_types(). Entities with no
    type at all are left out of the result.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        for entity_type in all_entity_types(entity):
            type_id = str(entity_type.id)
            group = groups.setdefault(type_id, {"entity_type": entity_type, "members": []})
            group["members"].append(entity)
    return groups


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
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Current description: {current_description or '(none)'}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {member.description}")
    return "\n".join(lines)


async def query_type_LLM(text_input, system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,
        response_model=EntityTypeDescription,
    )


def batch_members(members: List[Entity], batch_size: int) -> List[List[Entity]]:
    return [members[i : i + batch_size] for i in range(0, len(members), batch_size)]


def build_type_merge_prompt(
    entity_type_name: str,
    total_member_count: int,
    partial_descriptions: List[str],
    max_named_members: int = MAX_NAMED_MEMBERS,
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
        lines.append(f"{index}. {partial}")
    return "\n".join(lines)


async def query_type_merge_LLM(text_input, merge_system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=merge_system_prompt,
        response_model=EntityTypeDescription,
    )


def build_is_a_only_prompt(
    entity_type_name: str,
    final_type_description: str,
    members: List[Entity],
    total_member_count: int,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Final type summary: {final_type_description}",
        f"Total member count: {total_member_count}",
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {member.description}")
    return "\n".join(lines)


async def query_is_a_only_LLM(text_input, is_a_system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=is_a_system_prompt,
        response_model=EntityIsATexts,
    )


async def generate_type_description(
    entity_type: EntityType,
    members: List[Entity],
    system_prompt: str,
    merge_system_prompt: str,
    is_a_system_prompt: str,
) -> EntityTypeDescription:
    """Summarize a type's members, batching and merging the description when
    there are too many for a single prompt. Callers always pass the type's
    full member list here - batching is an internal detail, not something the
    caller decides.

    is_a lines are ALWAYS generated in their own call, separate from the
    description, never alongside it. The two jobs need contradictory naming
    rules whenever there are more than MAX_NAMED_MEMBERS members: the
    description must not name anyone, while every is_a line must start with a
    member's name - one response can't honor both at once. Beyond that
    correctness issue, a per-batch partial description also only sees a
    fraction of the members, so a comparative claim ("handles the most
    packages") could be true for that batch and false once every member is
    considered - another reason is_a lines must wait for the final,
    already-merged description before being generated.
    """
    total_member_count = len(members)
    batches = batch_members(members, MAX_MEMBERS_PER_TYPE_PROMPT)

    if len(batches) == 1:
        text = build_entity_type_prompt(
            entity_type.name, entity_type.description, members, total_member_count
        )
        result = await query_type_LLM(text, system_prompt)
        final_description = result.description
    else:
        partial_results = await asyncio.gather(
            *(
                query_type_LLM(
                    build_entity_type_prompt(
                        entity_type.name, entity_type.description, batch, total_member_count
                    ),
                    system_prompt,
                )
                for batch in batches
            )
        )
        partial_descriptions = [result.description for result in partial_results]
        merge_text = build_type_merge_prompt(
            entity_type.name, total_member_count, partial_descriptions
        )
        merged = await query_type_merge_LLM(merge_text, merge_system_prompt)
        final_description = merged.description

    is_a_results = await asyncio.gather(
        *(
            query_is_a_only_LLM(
                build_is_a_only_prompt(
                    entity_type.name, final_description, batch, total_member_count
                ),
                is_a_system_prompt,
            )
            for batch in batches
        )
    )
    is_a_texts = [text for result in is_a_results for text in result.is_a_texts]

    return EntityTypeDescription(description=final_description, is_a_texts=is_a_texts)


def apply_type_description(
    entity_type: EntityType,
    members: List[Entity],
    new_description: str,
    is_a_texts: Optional[List[MemberIsAText]] = None,
) -> EntityType:
    """Build one updated EntityType (same id, all other fields preserved) and
    point every member's is_a at that same shared instance.

    A single shared instance per type - not a copy per entity - is required:
    get_graph_from_model dedupes nested DataPoints by id and keeps only the
    first one it encounters, so independent per-entity copies of the "same"
    EntityType would silently lose all but one member's write.

    When a member has a matching is_a_text, is_a becomes the
    (Edge(relationship_type="is_a", edge_text=...), updated_entity_type) tuple
    so the text is searchable on the edge. A member with no matching text
    (name mismatch, or none produced) falls back to the bare EntityType rather
    than raising - the entity is still rewritten, just without edge_text.

    A member with more than one type appears here once per type it belongs to
    (once per call to this function, across different groups - see
    group_entities_by_type). Each call must only touch THIS type's slot -
    is_a if this is the member's sole type, or the matching tuple inside
    relations otherwise - and leave the member's other types exactly as they
    were, since a later call for another of its types still needs them intact.
    """
    updated_entity_type = entity_type.model_copy(update={"description": new_description})
    is_a_text_by_name = {item.member_name: item.is_a_text for item in (is_a_texts or [])}

    for member in members:
        is_a_text = is_a_text_by_name.get(member.name)

        primary_type = _entity_type_of(member.is_a)
        if primary_type is not None and primary_type.id == entity_type.id:
            # is_a is a scalar field: get_graph_from_model derives the "is_a"
            # relationship name from the field name itself when there's no
            # Edge wrapper, so a bare EntityType here still persists correctly.
            member.is_a = (
                (Edge(relationship_type="is_a", edge_text=is_a_text), updated_entity_type)
                if is_a_text
                else updated_entity_type
            )
            continue

        for index, relation in enumerate(member.relations):
            if _is_a_relation_type(relation) is not None and relation[1].id == entity_type.id:
                # relations is a list field: without an explicit Edge wrapper,
                # get_graph_from_model would label this edge "relations"
                # instead of "is_a" (it falls back to the field name). Always
                # wrap here, even with edge_text=None, to keep the "is_a"
                # label - unlike the is_a slot above, there is no bare form.
                member.relations[index] = (
                    Edge(relationship_type="is_a", edge_text=is_a_text),
                    updated_entity_type,
                )
                break

    return updated_entity_type


async def generate_type_descriptions(entities: List[Entity]) -> List[Entity]:
    """Group rewritten entities by type, generate one type description per group,
    and point every member's is_a at the shared updated EntityType.

    Entities with no type pass through untouched. Returns the same list of
    entities (now with is_a updated where applicable), ready for add_data_points.
    """
    groups = group_entities_by_type(entities)
    system_prompt = render_prompt(type_prompt_name, {})
    merge_system_prompt = render_prompt(type_merge_prompt_name, {})
    is_a_system_prompt = render_prompt(is_a_only_prompt_name, {})
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TYPE_LLM_CALLS)

    async def process_group(group: Dict[str, Any]) -> None:
        async with semaphore:
            entity_type = group["entity_type"]
            members = group["members"]
            result = await generate_type_description(
                entity_type, members, system_prompt, merge_system_prompt, is_a_system_prompt
            )
            apply_type_description(entity_type, members, result.description, result.is_a_texts)

    await asyncio.gather(*(process_group(group) for group in groups.values()))

    return entities
