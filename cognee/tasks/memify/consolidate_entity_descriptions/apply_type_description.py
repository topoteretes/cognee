from typing import Any

from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity
from cognee.shared.logging_utils import get_logger

from .constants import MAX_PERSISTED_IS_A_CHARS, truncate
from .models import MemberIsAText
from .type_links import iter_type_links, update_type_link

logger = get_logger("consolidate_entity_descriptions")


def group_entities_by_type(entities: list[Entity]) -> dict[str, dict[str, Any]]:
    """Group rewritten entities by their EntityType id.

    An entity with multiple types is registered as a member of every one of
    its type groups, not just one - see type_links.iter_type_links(). Entities
    with no type at all are left out of the result.
    """
    groups: dict[str, dict[str, Any]] = {}
    for entity in entities:
        for entity_type in iter_type_links(entity):
            type_id = str(entity_type.id)
            group = groups.setdefault(type_id, {"entity_type": entity_type, "members": []})
            group["members"].append(entity)
    return groups


def apply_type_description(
    entity_type: EntityType,
    members: list[Entity],
    new_description: str,
    is_a_texts: list[MemberIsAText] | None = None,
    max_persisted_is_a_chars: int = MAX_PERSISTED_IS_A_CHARS,
) -> EntityType:
    """Build one updated EntityType (same id, all other fields preserved) and
    point every member's is_a at that same shared instance.

    A single shared instance per type - not a copy per entity - is required:
    get_graph_from_model dedupes nested DataPoints by id and keeps only the
    first one it encounters, so independent per-entity copies of the "same"
    EntityType would silently lose all but one member's write.

    When a member has a matching is_a_text, is_a becomes the
    (Edge(relationship_type="is_a", edge_text=...), updated_entity_type) tuple
    so the text is searchable on the edge - truncated to max_persisted_is_a_chars
    before it's persisted, regardless of what the LLM call's own output
    budget let through. A member with no matching text (name mismatch, or
    none produced) falls back to the bare EntityType rather than raising -
    the entity is still rewritten, just without edge_text.
    Misses are counted and logged once per type (not raised) so a mismatch is
    visible instead of silently indistinguishable from a full match - the
    graph stays populated either way via prepare_edges_for_storage's generic
    edge_text fallback, so nothing else would ever surface it.

    A member with more than one type appears here once per type it belongs to
    (once per call to this function, across different groups - see
    group_entities_by_type). type_links.update_type_link() is what keeps each
    call to this type's own slot, leaving the member's other types intact for
    the later call that handles them.
    """
    updated_entity_type = entity_type.model_copy(update={"description": new_description})
    is_a_text_by_name = {item.member_name: item.is_a_text for item in (is_a_texts or [])}
    missed_count = 0

    for member in members:
        is_a_text = is_a_text_by_name.get(member.name)
        if is_a_text:
            # Bounds what actually gets persisted, independent of the output
            # token budget on the LLM call that produced it - that budget caps
            # generation, this caps what's written to the graph afterward.
            is_a_text = truncate(is_a_text, max_persisted_is_a_chars)
        else:
            missed_count += 1

        update_type_link(member, updated_entity_type, is_a_text)

    if missed_count > 0:
        logger.warning(
            "apply_type_description: %d of %d members for %r got no is_a line",
            missed_count,
            len(members),
            entity_type.name,
        )

    return updated_entity_type
