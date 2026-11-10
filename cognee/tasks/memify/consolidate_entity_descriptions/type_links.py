"""The one owner of how an Entity's types are stored on the in-memory node.

An entity's types live in two slots:

- ``is_a`` holds the first type, either bare (``EntityType``) or wrapped as
  ``(Edge(relationship_type="is_a", edge_text=...), EntityType)``.
- every additional type sits in ``relations`` as an is_a-tagged tuple.

Two slots rather than one list because ``is_a`` must never be empty for an
entity that has a type: ``get_graph_from_model`` derives the relationship name
from the scalar field itself, so a bare ``EntityType`` there still persists as
an ``is_a`` edge. ``relations`` is a list field, so an entry there without an
explicit ``Edge`` would be labelled ``"relations"`` - entries there always
carry one, even when ``edge_text`` is None. Extra types are not "lesser" than
the one on ``is_a``, just not the one exposed on the scalar field other code
already reads. Same convention ``rdf_ingest.py`` uses for ontology individuals
with more than one ``rdf:type``.

This module exists because that rule used to be written in ``build_entity``,
read back in ``all_entity_types``, and written a third time in
``apply_type_description``. The reader drifted from the writer once already
and silently dropped every type after the first.
"""

from collections.abc import Iterator
from typing import Any

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity


def entity_type_of(is_a: EntityType | tuple | None) -> EntityType | None:
    """Unwrap an is_a slot to its EntityType, whether bare or (Edge, EntityType)."""
    if isinstance(is_a, tuple):
        return is_a[1]
    return is_a


def is_a_relation_type(relation: Any) -> EntityType | None:
    """Return the EntityType of an is_a-tagged (Edge, EntityType) entry in relations, else None."""
    if (
        isinstance(relation, tuple)
        and len(relation) == 2
        and isinstance(relation[0], Edge)
        and relation[0].relationship_type == "is_a"
        and isinstance(relation[1], EntityType)
    ):
        return relation[1]
    return None


def set_type_links(entity: Entity, entity_types: list[EntityType]) -> None:
    """Write the convention: first type on is_a, the rest on relations.

    Both slots are always assigned, so an entity rebuilt with no types has
    them explicitly cleared rather than inheriting whatever was there.
    """
    entity.is_a = entity_types[0] if entity_types else None
    entity.relations = [
        (Edge(relationship_type="is_a"), entity_type) for entity_type in entity_types[1:]
    ]


def iter_type_links(entity: Entity) -> Iterator[EntityType]:
    """Every type this entity belongs to - the one on is_a plus the extras on relations.

    Must combine both, not treat them as alternatives: is_a is always
    populated when the entity has a type, so stopping as soon as it is found
    would drop every extra type for a multi-type entity.
    """
    primary = entity_type_of(entity.is_a)
    if primary is not None:
        yield primary
    for relation in entity.relations:
        entity_type = is_a_relation_type(relation)
        if entity_type is not None:
            yield entity_type


def update_type_link(entity: Entity, entity_type: EntityType, edge_text: str | None) -> bool:
    """Point this entity's link to entity_type.id at that instance, with optional edge text.

    Touches only the slot holding this type and leaves the entity's other
    types exactly as they were - a multi-type member is updated once per type
    it belongs to, and each call still needs the others intact.

    Returns False when the entity has no link to this type, so a caller that
    expected one can say so instead of silently doing nothing.
    """
    edge = Edge(relationship_type="is_a", edge_text=edge_text) if edge_text else None

    primary_type = entity_type_of(entity.is_a)
    if primary_type is not None and primary_type.id == entity_type.id:
        # is_a is a scalar field: get_graph_from_model derives the "is_a"
        # relationship name from the field name itself, so a bare EntityType
        # here still persists correctly.
        entity.is_a = (edge, entity_type) if edge else entity_type
        return True

    for index, relation in enumerate(entity.relations):
        if is_a_relation_type(relation) is not None and relation[1].id == entity_type.id:
            # relations is a list field: without an explicit Edge wrapper this
            # edge would be labelled "relations" (the field name). Always wrap,
            # even with edge_text=None - unlike is_a, there is no bare form.
            entity.relations[index] = (
                edge or Edge(relationship_type="is_a"),
                entity_type,
            )
            return True

    return False
