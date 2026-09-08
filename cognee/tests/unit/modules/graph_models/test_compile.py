import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph_models import (
    GraphSchemaSpec,
    graph_model_from_spec,
    graph_spec_to_json_schema,
)
from cognee.shared.graph_model_utils import graph_model_to_graph_schema

PEOPLE_SPEC = {
    "root": "Person",
    "entities": [
        {
            "name": "Person",
            "description": "A person.",
            "fields": [
                {"kind": "primitive", "name": "age", "primitive_type": "number"},
                {"kind": "enum", "name": "mood", "enum_values": ["happy", "grumpy"]},
                {
                    "kind": "relation",
                    "name": "works_at",
                    "relation": {"target_entity_name": "Organization", "cardinality": "one"},
                },
                {
                    "kind": "relation",
                    "name": "knows",
                    "relation": {"target_entity_name": "Person", "cardinality": "many"},
                },
            ],
        },
        {"name": "Organization", "identity_fields": ["name"], "fields": []},
    ],
}


def test_schema_shape_matches_frontend_compiler():
    schema = graph_spec_to_json_schema(PEOPLE_SPEC)

    # Root entity spread at top level with its title (required by the converter).
    assert schema["title"] == "Person"
    assert schema["type"] == "object"
    assert schema["required"] == ["name", "is_type"]

    # Every entity and its type marker registered in $defs.
    assert set(schema["$defs"]) == {"Person", "PersonType", "Organization", "OrganizationType"}

    person = schema["$defs"]["Person"]
    assert person["properties"]["is_type"] == {"$ref": "#/$defs/PersonType"}
    assert person["properties"]["age"] == {"type": "number"}
    assert person["properties"]["mood"] == {"enum": ["happy", "grumpy"], "type": "string"}
    # Cardinality one -> plain $ref; many -> array of $ref with [] default.
    assert person["properties"]["works_at"] == {"$ref": "#/$defs/Organization"}
    assert person["properties"]["knows"] == {
        "default": [],
        "items": {"$ref": "#/$defs/Person"},
        "type": "array",
    }

    marker = schema["$defs"]["PersonType"]
    assert marker["properties"]["name"]["default"] == "Person"


def test_identity_fields_emitted_in_metadata_defaults():
    schema = graph_spec_to_json_schema(PEOPLE_SPEC)

    metadata_default = schema["$defs"]["Person"]["properties"]["metadata"]["default"]
    assert metadata_default["index_fields"] == ["name"]
    assert metadata_default["identity_fields"] == ["name"]  # Python-side default

    org_default = schema["$defs"]["Organization"]["properties"]["metadata"]["default"]
    assert org_default["identity_fields"] == ["name"]


def test_identity_fields_opt_out():
    spec = {
        "entities": [
            {"name": "Ephemeral", "identity_fields": [], "fields": []},
        ]
    }
    schema = graph_spec_to_json_schema(spec)
    assert schema["properties"]["metadata"]["default"]["identity_fields"] == []


def test_accepts_model_or_dict():
    validated = GraphSchemaSpec.model_validate(PEOPLE_SPEC)
    assert graph_spec_to_json_schema(validated) == graph_spec_to_json_schema(PEOPLE_SPEC)


def test_round_trip_through_generated_model():
    model = graph_model_from_spec(PEOPLE_SPEC)

    assert isinstance(model, type)
    assert issubclass(model, DataPoint)
    assert model.__name__ == "Person"

    instance = model.model_validate(
        {
            "name": "Ada",
            "age": 36,
            "is_type": {},
            "works_at": {"name": "Analytical Engines", "is_type": {}},
            "knows": [],
        }
    )
    assert instance.name == "Ada"
    assert instance.works_at.name == "Analytical Engines"
    assert instance.metadata["index_fields"] == ["name"]
    assert instance.metadata["identity_fields"] == ["name"]

    # Inverse direction still produces a schema with the same title.
    regenerated = graph_model_to_graph_schema(model)
    assert regenerated["title"] == "Person"


def test_identity_fields_drive_deterministic_ids():
    model = graph_model_from_spec(PEOPLE_SPEC)

    first = model.model_validate({"name": "Ada", "is_type": {}})
    second = model.model_validate({"name": "Ada", "is_type": {}})
    third = model.model_validate({"name": "Grace", "is_type": {}})

    assert first.id == second.id
    assert first.id != third.id


def test_compile_rejects_invalid_spec():
    with pytest.raises(Exception):
        graph_spec_to_json_schema({"entities": [{"name": "bad name!"}]})
