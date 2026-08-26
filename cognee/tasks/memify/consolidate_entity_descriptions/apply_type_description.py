from typing import Any, Dict, List, Optional, Union

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity
from cognee.shared.logging_utils import get_logger

from .constants import MAX_TYPE_TEXT_CHARS
from .models import MemberIsAText

logger = get_logger("consolidate_entity_descriptions")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


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
    """Every type this entity belongs to - the one on is_a plus any extras on
    relations (see build_entity). Must combine both, not treat them as
    alternatives: is_a is now always populated when the entity has a type, so
    stopping as soon as it's found would silently drop every extra type for a
    multi-type entity - the exact bug this pipeline exists to fix."""
    primary = _entity_type_of(entity.is_a)
    from_relations = [
        entity_type
        for relation in entity.relations
        if (entity_type := _is_a_relation_type(relation)) is not None
    ]
    return ([primary] if primary is not None else []) + from_relations


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


def apply_type_description(
    entity_type: EntityType,
    members: List[Entity],
    new_description: str,
    is_a_texts: Optional[List[MemberIsAText]] = None,
    max_type_text_chars: int = MAX_TYPE_TEXT_CHARS,
) -> EntityType:
    """Build one updated EntityType (same id, all other fields preserved) and
    point every member's is_a at that same shared instance.

    A single shared instance per type - not a copy per entity - is required:
    get_graph_from_model dedupes nested DataPoints by id and keeps only the
    first one it encounters, so independent per-entity copies of the "same"
    EntityType would silently lose all but one member's write.

    When a member has a matching is_a_text, is_a becomes the
    (Edge(relationship_type="is_a", edge_text=...), updated_entity_type) tuple
    so the text is searchable on the edge - truncated to max_type_text_chars
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
    group_entities_by_type). Each call must only touch THIS type's slot -
    is_a if this is the member's sole type, or the matching tuple inside
    relations otherwise - and leave the member's other types exactly as they
    were, since a later call for another of its types still needs them intact.
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
            is_a_text = _truncate(is_a_text, max_type_text_chars)
        else:
            missed_count += 1

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

    if missed_count > 0:
        logger.warning(
            "apply_type_description: %d of %d members for %r got no is_a line",
            missed_count,
            len(members),
            entity_type.name,
        )

    return updated_entity_type
