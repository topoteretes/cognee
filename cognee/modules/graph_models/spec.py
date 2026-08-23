"""
Pydantic spec models for the JSON graph-model DSL.

A ``GraphSchemaSpec`` is the friendly, hand-writable JSON shape for declaring a
custom graph model — entities, typed fields, and relations with cardinality —
mirroring the frontend graph-model editor
(``cognee-frontend/src/modules/graphModels/types.ts``). It compiles (via
``cognee.modules.graph_models.compiler.graph_spec_to_json_schema``) into the JSON
Schema accepted by ``cognee.shared.graph_model_utils.graph_schema_to_graph_model``,
which generates a DataPoint-derived Pydantic model usable as ``graph_model`` in
``cognify()`` / ``remember()``.

Field names are snake_case with camelCase aliases (``populate_by_name=True``),
so one JSON document works for both the frontend editor and Python.

Validation here is also the safety gate: the downstream converter ``exec``s
code generated from the schema, so entity and field names are restricted to
plain identifiers and overall size is capped.
"""

import re
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cognee.infrastructure.engine import DataPoint

PrimitiveType = Literal["string", "number", "boolean", "date"]
Cardinality = Literal["one", "many"]

MAX_ENTITIES = 50
MAX_FIELDS_PER_ENTITY = 40

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Names owned by the DataPoint infrastructure (id, metadata, provenance stamps, ...)
# plus the generated type-marker field. User fields may not collide with these —
# the generated model inherits them from DataPoint.
RESERVED_FIELD_NAMES = frozenset(DataPoint.model_fields) | {"is_type"}


class _SpecBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class InverseSpec(_SpecBase):
    """Reserved: accepted for frontend parity but not compiled into the schema."""

    enabled: bool = True
    name: Optional[str] = None
    cardinality: Optional[Cardinality] = None


class RelationSpec(_SpecBase):
    target_entity_name: str = Field(alias="targetEntityName")
    cardinality: Cardinality = "one"
    inverse: Optional[InverseSpec] = None


class PrimitiveFieldSpec(_SpecBase):
    kind: Literal["primitive"] = "primitive"
    name: str
    primitive_type: PrimitiveType = Field(default="string", alias="primitiveType")
    required: bool = False
    description: Optional[str] = None


class EnumFieldSpec(_SpecBase):
    kind: Literal["enum"] = "enum"
    name: str
    enum_values: list[str] = Field(alias="enumValues", min_length=1)
    required: bool = False
    description: Optional[str] = None


class RelationFieldSpec(_SpecBase):
    kind: Literal["relation"] = "relation"
    name: str
    relation: RelationSpec
    required: bool = False
    description: Optional[str] = None


FieldSpec = Annotated[
    Union[PrimitiveFieldSpec, EnumFieldSpec, RelationFieldSpec],
    Field(discriminator="kind"),
]


class EntitySpec(_SpecBase):
    name: str
    description: Optional[str] = None
    primary_label_field: Optional[str] = Field(default=None, alias="primaryLabelField")
    index_fields: Optional[list[str]] = Field(default=None, alias="indexFields")
    # Fields whose values determine node identity: two extracted nodes with equal
    # identity-field values merge into one graph node. Defaults to ["name"] at
    # compile time; set to [] to opt out of merging (every node gets a random id).
    # NOTE: this is a Python-side extension — the frontend compiler never emits
    # identity_fields, so frontend-built models do not merge nodes.
    identity_fields: Optional[list[str]] = Field(default=None, alias="identityFields")
    fields: list[FieldSpec] = Field(default_factory=list)


class GraphSchemaOptions(_SpecBase):
    auto_type_nodes: bool = Field(default=True, alias="autoTypeNodes")


class GraphSchemaSpec(_SpecBase):
    options: GraphSchemaOptions = GraphSchemaOptions()
    # Entity used as the top-level model; defaults to the first entity.
    root: Optional[str] = None
    entities: list[EntitySpec] = Field(min_length=1, max_length=MAX_ENTITIES)

    def root_entity(self) -> EntitySpec:
        if self.root is None:
            return self.entities[0]
        return next(entity for entity in self.entities if entity.name == self.root)

    @model_validator(mode="after")
    def _validate_spec(self) -> "GraphSchemaSpec":
        entity_names = [entity.name for entity in self.entities]
        entity_name_set = set(entity_names)

        if len(entity_name_set) != len(entity_names):
            raise ValueError("Entity names must be unique.")

        for name in entity_names:
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(
                    f"Entity name {name!r} must be a plain identifier "
                    "(letters, digits, underscores; starting with a letter)."
                )
            # The compiler emits a "{Name}Type" marker definition per entity.
            if name.endswith("Type") and name[: -len("Type")] in entity_name_set:
                raise ValueError(
                    f"Entity name {name!r} collides with the generated type marker "
                    f"of entity {name[: -len('Type')]!r}."
                )

        if self.root is not None and self.root not in entity_name_set:
            raise ValueError(f"Root entity {self.root!r} is not declared.")

        for entity in self.entities:
            self._validate_entity(entity, entity_name_set)

        return self

    @staticmethod
    def _validate_entity(entity: EntitySpec, entity_name_set: set) -> None:
        if len(entity.fields) > MAX_FIELDS_PER_ENTITY:
            raise ValueError(
                f"Entity {entity.name!r} declares {len(entity.fields)} fields; "
                f"the maximum is {MAX_FIELDS_PER_ENTITY}."
            )

        field_names = [field.name for field in entity.fields]
        if len(set(field_names)) != len(field_names):
            raise ValueError(f"Entity {entity.name!r} has duplicate field names.")

        value_field_names = {"name"}
        for field in entity.fields:
            if not _IDENTIFIER_RE.match(field.name):
                raise ValueError(f"Field {entity.name}.{field.name!r} must be a plain identifier.")
            if field.name in RESERVED_FIELD_NAMES:
                raise ValueError(
                    f"Field {entity.name}.{field.name!r} collides with a DataPoint "
                    "infrastructure field."
                )
            if field.name == "name":
                # `name` is always emitted by the compiler as a required string;
                # an explicit declaration is only allowed as that exact shape.
                if field.kind != "primitive" or field.primitive_type != "string":
                    raise ValueError(
                        f"Field {entity.name}.name must be a string primitive "
                        "(it is the node's primary identifier)."
                    )
                continue
            if field.kind == "relation":
                target = field.relation.target_entity_name
                if target not in entity_name_set:
                    raise ValueError(
                        f"Relation {entity.name}.{field.name} targets undeclared entity {target!r}."
                    )
            else:
                value_field_names.add(field.name)

        for attribute in ("index_fields", "identity_fields"):
            referenced = getattr(entity, attribute)
            for field_name in referenced or []:
                if field_name not in value_field_names:
                    raise ValueError(
                        f"{attribute} entry {field_name!r} on entity {entity.name!r} "
                        "must reference 'name' or a declared primitive/enum field."
                    )
        if entity.primary_label_field and entity.primary_label_field not in value_field_names:
            raise ValueError(
                f"primary_label_field {entity.primary_label_field!r} on entity "
                f"{entity.name!r} must reference 'name' or a declared primitive/enum field."
            )
