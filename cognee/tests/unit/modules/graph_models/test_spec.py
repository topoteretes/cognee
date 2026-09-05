import pytest
from pydantic import ValidationError

from cognee.modules.graph_models import GraphSchemaSpec


def make_spec(**overrides):
    spec = {
        "entities": [
            {
                "name": "Person",
                "fields": [
                    {"kind": "primitive", "name": "age", "primitive_type": "number"},
                    {
                        "kind": "relation",
                        "name": "works_at",
                        "relation": {"target_entity_name": "Organization"},
                    },
                ],
            },
            {"name": "Organization", "fields": []},
        ],
    }
    spec.update(overrides)
    return spec


def test_valid_spec_parses():
    spec = GraphSchemaSpec.model_validate(make_spec())
    assert spec.root_entity().name == "Person"
    assert spec.entities[0].fields[1].relation.cardinality == "one"


def test_camel_case_aliases_accepted():
    spec = GraphSchemaSpec.model_validate(
        {
            "entities": [
                {
                    "name": "Person",
                    "indexFields": ["name"],
                    "identityFields": ["name"],
                    "fields": [
                        {"kind": "primitive", "name": "age", "primitiveType": "number"},
                        {
                            "kind": "relation",
                            "name": "works_at",
                            "relation": {"targetEntityName": "Organization", "cardinality": "many"},
                        },
                    ],
                },
                {"name": "Organization"},
            ]
        }
    )
    assert spec.entities[0].fields[0].primitive_type == "number"
    assert spec.entities[0].fields[1].relation.target_entity_name == "Organization"


def test_unknown_relation_target_rejected():
    spec = make_spec()
    spec["entities"][0]["fields"][1]["relation"]["target_entity_name"] = "Nowhere"
    with pytest.raises(ValidationError, match="undeclared"):
        GraphSchemaSpec.model_validate(spec)


def test_duplicate_entity_names_rejected():
    spec = make_spec()
    spec["entities"].append({"name": "Person"})
    with pytest.raises(ValidationError, match="unique"):
        GraphSchemaSpec.model_validate(spec)


def test_duplicate_field_names_rejected():
    spec = make_spec()
    spec["entities"][0]["fields"].append(
        {"kind": "primitive", "name": "age", "primitive_type": "string"}
    )
    with pytest.raises(ValidationError, match="duplicate field names"):
        GraphSchemaSpec.model_validate(spec)


@pytest.mark.parametrize(
    "hostile_name",
    ['"; import os', "__class__", "1starts_with_digit", "has space", "has-dash", ""],
)
def test_hostile_entity_names_rejected(hostile_name):
    spec = make_spec()
    spec["entities"][1]["name"] = hostile_name
    with pytest.raises(ValidationError):
        GraphSchemaSpec.model_validate(spec)


@pytest.mark.parametrize(
    "hostile_name",
    ['"; import os', "__class__", "1x", "a b"],
)
def test_hostile_field_names_rejected(hostile_name):
    spec = make_spec()
    spec["entities"][0]["fields"][0]["name"] = hostile_name
    with pytest.raises(ValidationError):
        GraphSchemaSpec.model_validate(spec)


@pytest.mark.parametrize("reserved", ["id", "metadata", "is_type", "source_pipeline"])
def test_infra_field_names_rejected(reserved):
    spec = make_spec()
    spec["entities"][0]["fields"][0]["name"] = reserved
    with pytest.raises(ValidationError, match="infrastructure"):
        GraphSchemaSpec.model_validate(spec)


def test_type_marker_collision_rejected():
    spec = make_spec()
    spec["entities"].append({"name": "PersonType"})
    with pytest.raises(ValidationError, match="type marker"):
        GraphSchemaSpec.model_validate(spec)


def test_unknown_root_rejected():
    with pytest.raises(ValidationError, match="not declared"):
        GraphSchemaSpec.model_validate(make_spec(root="Ghost"))


def test_explicit_root_selected():
    spec = GraphSchemaSpec.model_validate(make_spec(root="Organization"))
    assert spec.root_entity().name == "Organization"


def test_index_fields_must_reference_value_fields():
    spec = make_spec()
    spec["entities"][0]["index_fields"] = ["works_at"]  # relation, not a value field
    with pytest.raises(ValidationError, match="index_fields"):
        GraphSchemaSpec.model_validate(spec)


def test_name_field_only_allowed_as_string_primitive():
    spec = make_spec()
    spec["entities"][0]["fields"][0] = {
        "kind": "primitive",
        "name": "name",
        "primitive_type": "number",
    }
    with pytest.raises(ValidationError, match="primary identifier"):
        GraphSchemaSpec.model_validate(spec)


def test_extra_keys_rejected():
    spec = make_spec()
    spec["entities"][0]["surprise"] = True
    with pytest.raises(ValidationError):
        GraphSchemaSpec.model_validate(spec)
