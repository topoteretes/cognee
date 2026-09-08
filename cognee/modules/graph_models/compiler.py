"""
Compiler from the JSON graph-model DSL (``GraphSchemaSpec``) to the JSON Schema
shape accepted by ``graph_schema_to_graph_model``.

This is a Python port of the frontend compiler
(``cognee-frontend/src/modules/graphModels/toGraphModelSchema.ts``) with one
deliberate extension: ``metadata`` defaults also carry ``identity_fields``
(default ``["name"]``), so nodes extracted from different chunks/runs with the
same identity values merge into one graph node. The frontend compiler emits
``index_fields`` only, which means frontend-built models never merge nodes.
Set ``identity_fields: []`` on an entity to opt out of merging.

Known composition caveats of custom graph models in general (not specific to
this DSL): extraction with a non-``KnowledgeGraph`` model bypasses ontology
grounding and the extra node/edge dedup passes in ``integrate_chunk_graphs``,
and does not compose with ``functional_relationships``.
"""

from typing import Union, cast

from .spec import EntitySpec, GraphSchemaSpec

_PRIMITIVE_TYPE_MAP = {
    "string": "string",
    "number": "number",
    "boolean": "boolean",
    "date": "string",  # ISO date string
}


def _index_fields(entity: EntitySpec) -> list:
    return entity.index_fields if entity.index_fields else ["name"]


def _identity_fields(entity: EntitySpec) -> list:
    return entity.identity_fields if entity.identity_fields is not None else ["name"]


def _metadata_property(entity: EntitySpec) -> dict:
    return {
        "additionalProperties": True,
        "default": {
            "index_fields": _index_fields(entity),
            "identity_fields": _identity_fields(entity),
        },
        "type": "object",
    }


def _build_entity_type_def(entity: EntitySpec) -> dict:
    # The "{Name}Type" marker every instance carries as `is_type`. Its `name`
    # defaults to the entity name and identity is by name, so all markers of one
    # entity type merge into a single graph node.
    return {
        "properties": {
            "name": {"default": entity.name, "type": "string"},
            "metadata": {
                "additionalProperties": True,
                "default": {
                    "index_fields": _index_fields(entity),
                    "identity_fields": ["name"],
                },
                "type": "object",
            },
        },
        "title": f"{entity.name}Type",
        "type": "object",
    }


def _build_entity_schema(entity: EntitySpec) -> dict:
    properties: dict = {
        # `name` is the primary node identifier — always present.
        "name": {"type": "string"},
        # `is_type` carries the entity type marker.
        "is_type": {"$ref": f"#/$defs/{entity.name}Type"},
        # `metadata` tells cognee which fields to index and merge on.
        "metadata": _metadata_property(entity),
    }
    required = ["name", "is_type"]

    for field in entity.fields:
        if field.name == "name":
            continue  # already added above

        if field.kind == "primitive":
            field_schema: dict = {"type": _PRIMITIVE_TYPE_MAP[field.primitive_type]}
        elif field.kind == "enum":
            field_schema = {"enum": list(field.enum_values), "type": "string"}
        else:  # relation
            target = field.relation.target_entity_name
            if field.relation.cardinality == "many":
                field_schema = {
                    "default": [],
                    "items": {"$ref": f"#/$defs/{target}"},
                    "type": "array",
                }
            else:
                field_schema = {"$ref": f"#/$defs/{target}"}

        if field.description:
            field_schema["description"] = field.description
        properties[field.name] = field_schema

        if field.kind != "relation" and field.required:
            required.append(field.name)

    entity_schema = {
        "properties": properties,
        "required": required,
        "title": entity.name,
        "type": "object",
    }
    if entity.description:
        entity_schema["description"] = entity.description
    return entity_schema


def graph_spec_to_json_schema(spec: Union[GraphSchemaSpec, dict]) -> dict:
    """Compile a DSL spec (model or plain dict) into a graph-model JSON Schema."""
    validated = spec if isinstance(spec, GraphSchemaSpec) else GraphSchemaSpec.model_validate(spec)

    defs: dict = {}
    # Register every entity and its type marker in $defs upfront; self-referential
    # relations are plain $ref pointers, so no recursion is needed.
    for entity in validated.entities:
        defs[f"{entity.name}Type"] = _build_entity_type_def(entity)
        defs[entity.name] = _build_entity_schema(entity)

    # The root entity's schema body is spread at the top level (its `title` is
    # what graph_schema_to_graph_model looks up in the generated module).
    return {"$defs": defs, **_build_entity_schema(validated.root_entity())}


def graph_model_from_spec(spec: Union[GraphSchemaSpec, dict]) -> type:
    """Validate a DSL spec, compile it, and generate the DataPoint-derived model."""
    # Imported lazily: graph_model_utils imports the cognee API surface for its
    # demo block, which would otherwise create an import cycle through low_level.
    from cognee.shared.graph_model_utils import graph_schema_to_graph_model

    # graph_schema_to_graph_model is annotated as returning an instance but
    # actually returns the generated model class.
    return cast(type, graph_schema_to_graph_model(graph_spec_to_json_schema(spec)))
