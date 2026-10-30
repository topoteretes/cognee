"""
Merge a partial graph-model patch into a stored ``GraphSchemaSpec`` document.

``PUT /datasets/{id}/schema`` replaces the stored spec wholesale; ``PATCH``
sends a ``GraphSchemaPatch`` and this module builds the merged document from
the stored spec. Entity and field names are the merge keys, mirroring the
frontend editor where a model is a list of named entities each holding a list
of named fields.

Order of operations: entity upserts, then field removals, then entity
removals, then a relation-target check, then the options shallow-merge. The
merged document is validated through ``GraphSchemaSpec`` at the end, so the
compiler's safety limits (identifier names, entity and field caps) still gate
everything that gets stored.
"""

from typing import Any

from pydantic import Field, ValidationError

from .spec import EntitySpec, FieldSpec, GraphSchemaOptions, GraphSchemaSpec, _SpecBase


class GraphSchemaMergeError(ValueError):
    """The patch cannot be applied to the stored spec (names the offending item)."""


class EntityPatch(_SpecBase):
    """Partial ``EntitySpec``.

    ``fields`` are upserted by name into the stored entity (a field whose name
    matches is replaced whole, whatever its kind). Scalars overwrite the stored
    value only when sent; an omitted key keeps the stored value, an explicit
    ``null`` clears it.
    """

    name: str
    description: str | None = None
    primary_label_field: str | None = Field(default=None, alias="primaryLabelField")
    index_fields: list[str] | None = Field(default=None, alias="indexFields")
    identity_fields: list[str] | None = Field(default=None, alias="identityFields")
    fields: list[FieldSpec] = Field(default_factory=list)


class FieldRemoval(_SpecBase):
    entity: str
    field: str


class GraphSchemaPatch(_SpecBase):
    """Body of ``PATCH /datasets/{id}/schema``; snake_case or camelCase keys."""

    entities: list[EntityPatch] = Field(default_factory=list)
    remove_entities: list[str] = Field(default_factory=list, alias="removeEntities")
    remove_fields: list[FieldRemoval] = Field(default_factory=list, alias="removeFields")
    # Shallow-merged over the stored options; only the keys sent change.
    options: GraphSchemaOptions | None = None
    # Overwrites the stored root when sent (an explicit null resets to the first entity).
    root: str | None = None


def format_validation_error(error: ValidationError) -> str:
    """One line per problem, without pydantic's ``Value error, `` prefix."""
    return " ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())


def _load_stored_entities(stored: dict[str, Any] | None) -> tuple[GraphSchemaSpec | None, list]:
    if not stored:
        return None, []
    try:
        spec = GraphSchemaSpec.model_validate(stored)
    except ValidationError as error:
        raise GraphSchemaMergeError(
            "The stored graph schema is not a valid graph-model spec and cannot be patched; "
            f"replace it with PUT. {format_validation_error(error)}"
        ) from error
    return spec, list(spec.entities)


def _upsert_fields(existing: list, incoming: list) -> list:
    merged = list(existing)
    positions = {field.name: index for index, field in enumerate(merged)}
    for field in incoming:
        if field.name in positions:
            merged[positions[field.name]] = field
        else:
            positions[field.name] = len(merged)
            merged.append(field)
    return merged


def _upsert_entity(entities: list, patch: EntityPatch) -> None:
    for index, existing in enumerate(entities):
        if existing.name != patch.name:
            continue
        updates = patch.model_dump(exclude_unset=True, exclude={"name", "fields"})
        updates["fields"] = _upsert_fields(existing.fields, patch.fields)
        entities[index] = existing.model_copy(update=updates)
        return
    entities.append(EntitySpec.model_validate(patch.model_dump(exclude_unset=True, by_alias=True)))


def _remove_field(entities: list, removal: FieldRemoval) -> None:
    for index, entity in enumerate(entities):
        if entity.name != removal.entity:
            continue
        remaining = [field for field in entity.fields if field.name != removal.field]
        if len(remaining) == len(entity.fields):
            raise GraphSchemaMergeError(
                f"Cannot remove field {removal.field!r}: entity {removal.entity!r} "
                "has no field with that name."
            )
        entities[index] = entity.model_copy(update={"fields": remaining})
        return
    raise GraphSchemaMergeError(
        f"Cannot remove field {removal.entity}.{removal.field}: "
        f"entity {removal.entity!r} is not in the schema."
    )


def _remove_entities(entities: list, names: list[str]) -> list:
    present = {entity.name for entity in entities}
    for name in names:
        if name not in present:
            raise GraphSchemaMergeError(f"Cannot remove entity {name!r}: it is not in the schema.")
    removed = set(names)
    remaining = [entity for entity in entities if entity.name not in removed]
    for entity in remaining:
        for field in entity.fields:
            if field.kind == "relation" and field.relation.target_entity_name in removed:
                raise GraphSchemaMergeError(
                    f"Cannot remove entity {field.relation.target_entity_name!r}: "
                    f"relation {entity.name}.{field.name} still targets it."
                )
    return remaining


def merge_graph_schema(stored: dict[str, Any] | None, patch: GraphSchemaPatch) -> dict[str, Any]:
    """Apply ``patch`` to ``stored`` and return the validated merged document.

    ``stored`` is the JSON document kept on the dataset (``None`` when no schema
    has been stored yet, in which case the patch is applied to an empty spec).
    The result is the canonical camelCase form of the merged ``GraphSchemaSpec``.

    Raises ``GraphSchemaMergeError`` for a patch that cannot be applied (unknown
    entity or field to remove, removal leaving a relation without its target,
    unparseable stored spec) and ``pydantic.ValidationError`` when the merged
    document violates the spec rules.
    """
    stored_spec, entities = _load_stored_entities(stored)

    for entity_patch in patch.entities:
        _upsert_entity(entities, entity_patch)
    for removal in patch.remove_fields:
        _remove_field(entities, removal)
    entities = _remove_entities(entities, patch.remove_entities)

    options = stored_spec.options.model_dump(by_alias=True) if stored_spec else {}
    if patch.options is not None:
        options.update(patch.options.model_dump(by_alias=True, exclude_unset=True))

    root = stored_spec.root if stored_spec else None
    if "root" in patch.model_fields_set:
        root = patch.root

    merged = GraphSchemaSpec.model_validate(
        {
            "options": options,
            "root": root,
            "entities": [
                entity.model_dump(by_alias=True, exclude_none=True) for entity in entities
            ],
        }
    )
    return merged.model_dump(by_alias=True, exclude_none=True)
