from typing import Annotated, get_args

import pytest
from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint, FromIdentity
from cognee.shared.graph_model_utils import (
    datapoint_model_to_basemodel,
    from_identity_fields,
    graph_model_to_graph_schema,
    graph_schema_to_graph_model,
)
from cognee.tasks.graph.exceptions import InvalidReferenceTypeError

DATAPOINT_INFRA_FIELDS = {
    "id",
    "created_at",
    "updated_at",
    "ontology_valid",
    "version",
    "topological_rank",
    "type",
    "belongs_to_set",
    "source_pipeline",
    "source_task",
    "source_node_set",
    "source_user",
    "source_content_hash",
    "feedback_weight",
    "importance_weight",
    "ontology_uri",
    "valid_to",
    "metadata",
}


def assert_dump_has_no_infra(data):
    """Recursively assert model_dump output contains no DataPoint infra keys."""
    if isinstance(data, dict):
        assert DATAPOINT_INFRA_FIELDS.isdisjoint(data)
        for value in data.values():
            assert_dump_has_no_infra(value)
    elif isinstance(data, list):
        for item in data:
            assert_dump_has_no_infra(item)


@pytest.fixture
def programming_language_models():
    class FieldType(DataPoint):
        name: str = "Field"
        metadata: dict = {"index_fields": ["name"]}

    class Field(DataPoint):
        name: str
        is_type: FieldType
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguageType(DataPoint):
        name: str = "Programming Language"

    class ProgrammingLanguage(DataPoint):
        name: str
        used_in: list[Field] = []
        is_type: ProgrammingLanguageType
        metadata: dict = {"index_fields": ["name"]}

    return {
        "FieldType": FieldType,
        "Field": Field,
        "ProgrammingLanguageType": ProgrammingLanguageType,
        "ProgrammingLanguage": ProgrammingLanguage,
    }


def test_graph_model_to_graph_schema_supports_datapoint_subclasses():
    class FieldType(DataPoint):
        name: str = "Field"

    class Field(DataPoint):
        name: str
        is_type: FieldType

    schema = graph_model_to_graph_schema(Field)

    assert schema["title"] == "Field"
    assert "name" in schema["properties"]
    assert "is_type" in schema["properties"]
    assert "id" not in schema["properties"]
    assert "version" not in schema["properties"]

    field_type_schema = schema["$defs"]["FieldType"]
    assert "id" not in field_type_schema["properties"]
    assert "version" not in field_type_schema["properties"]


def test_graph_model_to_graph_schema_keeps_basemodel_behavior():
    class FieldType(BaseModel):
        name: str = "Field"

    class Field(BaseModel):
        name: str
        is_type: FieldType

    schema = graph_model_to_graph_schema(Field)

    assert schema["title"] == "Field"
    assert "name" in schema["properties"]
    assert "is_type" in schema["properties"]


def test_datapoint_model_to_basemodel_simplifies_single_class(programming_language_models):
    Field = programming_language_models["Field"]

    simplified = datapoint_model_to_basemodel(Field)

    assert issubclass(simplified, BaseModel)
    assert not issubclass(simplified, DataPoint)
    assert set(simplified.model_fields) == {"name", "is_type"}
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(simplified.model_fields)

    field_type = simplified.model_fields["is_type"].annotation
    assert field_type().name == "Field"

    instance = simplified(name="numpy", is_type=field_type())
    assert instance.name == "numpy"


def test_datapoint_model_to_basemodel_recurses_nested_types(programming_language_models):
    ProgrammingLanguage = programming_language_models["ProgrammingLanguage"]

    simplified = datapoint_model_to_basemodel(ProgrammingLanguage)
    field_type = simplified.model_fields["is_type"].annotation
    field_model = get_args(simplified.model_fields["used_in"].annotation)[0]
    nested_field_type = field_model.model_fields["is_type"].annotation

    assert set(simplified.model_fields) == {"name", "used_in", "is_type"}
    assert not issubclass(field_model, DataPoint)
    assert set(field_model.model_fields) == {"name", "is_type"}
    assert not issubclass(field_type, DataPoint)
    assert set(field_type.model_fields) == {"name"}
    assert not issubclass(nested_field_type, DataPoint)
    assert set(nested_field_type.model_fields) == {"name"}

    instance = simplified(
        name="Python",
        is_type=field_type(),
        used_in=[field_model(name="data analysis", is_type=nested_field_type())],
    )
    dump = instance.model_dump()

    assert dump["name"] == "Python"
    assert dump["used_in"][0]["name"] == "data analysis"
    assert_dump_has_no_infra(dump)


def test_datapoint_model_to_basemodel_passthrough():
    class PlainModel(BaseModel):
        name: str

    assert datapoint_model_to_basemodel(PlainModel) is PlainModel


def test_strip_metadata_flag(programming_language_models):
    Field = programming_language_models["Field"]
    ProgrammingLanguage = programming_language_models["ProgrammingLanguage"]

    # DataPoint.metadata is infrastructure, so it is already excluded by default.
    default_simplified = datapoint_model_to_basemodel(Field)
    assert "metadata" not in default_simplified.model_fields

    stripped = datapoint_model_to_basemodel(ProgrammingLanguage, strip_metadata=True)
    field_model = get_args(stripped.model_fields["used_in"].annotation)[0]
    field_type = stripped.model_fields["is_type"].annotation
    nested_field_type = field_model.model_fields["is_type"].annotation

    assert "metadata" not in stripped.model_fields
    assert "metadata" not in field_model.model_fields
    assert "metadata" not in field_type.model_fields
    assert "metadata" not in nested_field_type.model_fields


def test_rehydration_with_strip_metadata(programming_language_models):
    ProgrammingLanguage = programming_language_models["ProgrammingLanguage"]

    simplified = datapoint_model_to_basemodel(ProgrammingLanguage, strip_metadata=True)
    field_model = get_args(simplified.model_fields["used_in"].annotation)[0]
    field_type = simplified.model_fields["is_type"].annotation
    nested_field_type = field_model.model_fields["is_type"].annotation

    simplified_tree = simplified(
        name="Python",
        is_type=field_type(),
        used_in=[field_model(name="data analysis", is_type=nested_field_type())],
    )

    rehydrated = ProgrammingLanguage.model_validate(simplified_tree.model_dump())

    assert isinstance(rehydrated, ProgrammingLanguage)
    assert rehydrated.name == "Python"
    assert isinstance(rehydrated.is_type, programming_language_models["ProgrammingLanguageType"])
    assert isinstance(rehydrated.used_in[0], programming_language_models["Field"])
    assert rehydrated.used_in[0].name == "data analysis"
    assert rehydrated.metadata["index_fields"] == ["name"]
    assert rehydrated.used_in[0].metadata["index_fields"] == ["name"]


def test_preserves_required_inherited_fields():
    class Animal(DataPoint):
        species: str

    class Dog(Animal):
        breed: str

    simplified = datapoint_model_to_basemodel(Dog, strip_metadata=True)

    assert set(simplified.model_fields) == {"species", "breed"}
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(simplified.model_fields)


def test_preserves_defaulted_inherited_fields():
    class Animal(DataPoint):
        species: str
        age: int = 0

    class Dog(Animal):
        breed: str

    simplified = datapoint_model_to_basemodel(Dog, strip_metadata=True)

    assert set(simplified.model_fields) == {"species", "age", "breed"}
    assert simplified.model_fields["age"].default == 0


def test_preserves_inherited_nested_datapoint_fields():
    class Habitat(DataPoint):
        climate: str

    class Animal(DataPoint):
        species: str
        habitat: Habitat

    class Dog(Animal):
        breed: str

    simplified = datapoint_model_to_basemodel(Dog, strip_metadata=True)
    habitat_model = simplified.model_fields["habitat"].annotation

    assert set(simplified.model_fields) == {"species", "habitat", "breed"}
    assert set(habitat_model.model_fields) == {"climate"}
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(simplified.model_fields)
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(habitat_model.model_fields)
    assert "metadata" not in simplified.model_fields
    assert "metadata" not in habitat_model.model_fields


def test_preserves_inherited_fields_in_cyclic_models():
    class LinkedNode(DataPoint):
        label: str

    class NodeA(LinkedNode):
        other: "NodeB | None" = None

    class NodeB(LinkedNode):
        other: NodeA | None = None

    NodeA.model_rebuild()

    simplified_a = datapoint_model_to_basemodel(NodeA, strip_metadata=True)
    nested_b = simplified_a.model_fields["other"].annotation
    # Optional[NodeB] → get the non-None arm
    nested_b_model = next(arg for arg in get_args(nested_b) if arg is not type(None))

    assert "label" in simplified_a.model_fields
    assert "other" in simplified_a.model_fields
    assert "label" in nested_b_model.model_fields
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(simplified_a.model_fields)
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(nested_b_model.model_fields)


def test_from_identity_field_renders_as_string():
    class Role(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class Person(DataPoint):
        name: str
        is_a: Annotated[Role, FromIdentity()]
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    simplified = datapoint_model_to_basemodel(Person, strip_metadata=True)
    schema = simplified.model_json_schema()
    assert schema["properties"]["is_a"]["type"] == "string"
    assert "$ref" not in schema["properties"]["is_a"]
    assert DATAPOINT_INFRA_FIELDS.isdisjoint(simplified.model_fields)
    assert_dump_has_no_infra(simplified(name="Alice", is_a="Student").model_dump())


def test_model_without_from_identity_is_unchanged(programming_language_models):
    ProgrammingLanguage = programming_language_models["ProgrammingLanguage"]
    simplified = datapoint_model_to_basemodel(ProgrammingLanguage, strip_metadata=True)
    assert set(simplified.model_fields) == {"name", "used_in", "is_type"}
    assert simplified.model_fields["is_type"].annotation is not str


def test_from_identity_target_without_identity_raises():
    class Role(DataPoint):
        name: str

    class Person(DataPoint):
        name: str
        is_a: Annotated[Role, FromIdentity()]

    with pytest.raises(InvalidReferenceTypeError, match="Role"):
        datapoint_model_to_basemodel(Person)


def test_from_identity_target_with_extra_required_field_raises():
    class Role(DataPoint):
        name: str
        level: int
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class Person(DataPoint):
        name: str
        is_a: Annotated[Role, FromIdentity()]

    with pytest.raises(InvalidReferenceTypeError, match="level"):
        datapoint_model_to_basemodel(Person)


def test_from_identity_schema_round_trip_drops_the_marker():
    class Role(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    class Person(DataPoint):
        name: str
        is_a: Annotated[Role, FromIdentity()]
        metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    schema = graph_model_to_graph_schema(Person)
    regenerated = graph_schema_to_graph_model(schema)
    assert from_identity_fields(regenerated) == {}
