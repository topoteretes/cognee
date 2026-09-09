"""Pytest matrix for ``graph_schema_to_graph_model`` (JSON Schema -> Pydantic model).

These tests are pure pydantic: no DB, no LLM, no network. They exercise the
JSON-Schema feature matrix the converter is expected to support (nested objects,
arrays incl. arrays of objects, ``$ref``/``$defs``, ``enum``,
``oneOf``/``anyOf``/``allOf``, required vs optional, ``format``,
``additionalProperties``, ``default`` values, integer vs number, nullable) plus
class-name sanitization and a ``schema -> model -> schema`` round-trip.

The heavy lifting is done by ``datamodel-code-generator``; this suite pins the
behavior of cognee's wrapper around it, especially the parts the wrapper owns:
using ``DataPoint`` as the base class and resolving the generated class name.
"""

import copy
import enum
import types as _types
from datetime import datetime, timezone
from typing import Union, get_args, get_origin
from uuid import UUID

import pytest
from pydantic import ValidationError

from cognee.infrastructure.engine import DataPoint
from cognee.shared.graph_model_utils import (
    graph_model_to_graph_schema,
    graph_schema_to_graph_model,
)

convert = graph_schema_to_graph_model

DATAPOINT_INFRA_FIELDS = set(DataPoint.model_fields)
NoneType = type(None)


# --------------------------------------------------------------------------- #
# annotation helpers                                                          #
# --------------------------------------------------------------------------- #
def is_optional(annotation) -> bool:
    return get_origin(annotation) in (Union, _types.UnionType) and NoneType in get_args(
        annotation
    )


def non_none_args(annotation) -> tuple:
    return tuple(arg for arg in get_args(annotation) if arg is not NoneType)


def core_type(annotation):
    """Strip an ``Optional``/``| None`` wrapper and return the inner annotation."""
    if is_optional(annotation):
        args = non_none_args(annotation)
        return args[0] if len(args) == 1 else Union[args]
    return annotation


def own_fields(model) -> dict:
    """Fields declared by the schema, excluding inherited DataPoint infra fields."""
    return {
        name: info
        for name, info in model.model_fields.items()
        if name not in DATAPOINT_INFRA_FIELDS
    }


# --------------------------------------------------------------------------- #
# base contract                                                               #
# --------------------------------------------------------------------------- #
def test_generated_model_subclasses_datapoint():
    """The whole point: the model must plug into cognee's graph engine."""
    model = convert({"title": "Person", "type": "object", "properties": {"name": {"type": "string"}}})
    assert issubclass(model, DataPoint)
    assert model.__name__ == "Person"


# --------------------------------------------------------------------------- #
# required vs optional                                                        #
# --------------------------------------------------------------------------- #
def test_required_vs_optional():
    model = convert(
        {
            "title": "Doc",
            "type": "object",
            "properties": {"title": {"type": "string"}, "subtitle": {"type": "string"}},
            "required": ["title"],
        }
    )
    fields = own_fields(model)
    assert not is_optional(fields["title"].annotation)
    assert is_optional(fields["subtitle"].annotation)


def test_missing_required_field_raises():
    model = convert(
        {
            "title": "Doc2",
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }
    )
    with pytest.raises(ValidationError):
        model()  # 'title' is required


# --------------------------------------------------------------------------- #
# scalar type mapping (integer vs number, boolean, string)                    #
# --------------------------------------------------------------------------- #
def test_integer_vs_number_are_distinct():
    model = convert(
        {
            "title": "Measure",
            "type": "object",
            "properties": {"count": {"type": "integer"}, "ratio": {"type": "number"}},
        }
    )
    fields = own_fields(model)
    assert core_type(fields["count"].annotation) is int
    assert core_type(fields["ratio"].annotation) is float


def test_boolean_and_string():
    model = convert(
        {
            "title": "Flagged",
            "type": "object",
            "properties": {"active": {"type": "boolean"}, "label": {"type": "string"}},
        }
    )
    fields = own_fields(model)
    assert core_type(fields["active"].annotation) is bool
    assert core_type(fields["label"].annotation) is str


# --------------------------------------------------------------------------- #
# nested objects & arrays                                                     #
# --------------------------------------------------------------------------- #
def test_nested_inline_object():
    model = convert(
        {
            "title": "Company",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ceo": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
    )
    ceo_type = core_type(own_fields(model)["ceo"].annotation)
    assert issubclass(ceo_type, DataPoint)
    # nested model is usable
    instance = model(name="Acme", ceo=ceo_type(name="Ada"))
    assert instance.ceo.name == "Ada"


def test_array_of_scalars():
    model = convert(
        {
            "title": "Tagged",
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
    )
    tags = core_type(own_fields(model)["tags"].annotation)
    assert get_origin(tags) is list
    assert get_args(tags)[0] is str


def test_array_of_objects():
    model = convert(
        {
            "title": "Team",
            "type": "object",
            "properties": {
                "members": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            },
        }
    )
    members = core_type(own_fields(model)["members"].annotation)
    assert get_origin(members) is list
    member_type = get_args(members)[0]
    assert issubclass(member_type, DataPoint)


# --------------------------------------------------------------------------- #
# $ref / $defs                                                                #
# --------------------------------------------------------------------------- #
def test_ref_and_defs_resolve():
    model = convert(
        {
            "title": "Order",
            "type": "object",
            "properties": {"customer": {"$ref": "#/$defs/Customer"}},
            "$defs": {"Customer": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
    )
    customer_type = core_type(own_fields(model)["customer"].annotation)
    assert issubclass(customer_type, DataPoint)
    assert customer_type.__name__ == "Customer"


# --------------------------------------------------------------------------- #
# enum                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status_schema",
    [
        {"enum": ["open", "closed"]},  # untyped enum
        {"type": "string", "enum": ["open", "closed"]},  # typed enum
    ],
)
def test_enum(status_schema):
    model = convert(
        {"title": "Ticket", "type": "object", "properties": {"status": status_schema}}
    )
    status_type = core_type(own_fields(model)["status"].annotation)
    assert issubclass(status_type, enum.Enum)
    assert {member.value for member in status_type} == {"open", "closed"}


# --------------------------------------------------------------------------- #
# format: date-time / uuid / email                                            #
# --------------------------------------------------------------------------- #
def test_format_date_time():
    model = convert(
        {
            "title": "Event",
            "type": "object",
            "properties": {"when": {"type": "string", "format": "date-time"}},
        }
    )
    instance = model(when=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert isinstance(instance.when, datetime)


def test_format_uuid():
    model = convert(
        {
            "title": "Ref",
            "type": "object",
            "properties": {"ref_id": {"type": "string", "format": "uuid"}},
        }
    )
    instance = model(ref_id="12345678-1234-5678-1234-567812345678")
    assert isinstance(instance.ref_id, UUID)


def test_format_email():
    pytest.importorskip("email_validator")  # optional dep behind EmailStr
    model = convert(
        {
            "title": "Contact",
            "type": "object",
            "properties": {"email": {"type": "string", "format": "email"}},
        }
    )
    assert model(email="a@b.com").email == "a@b.com"
    with pytest.raises(ValidationError):
        model(email="not-an-email")


# --------------------------------------------------------------------------- #
# defaults                                                                     #
# --------------------------------------------------------------------------- #
def test_default_values_are_preserved_and_applied():
    model = convert(
        {
            "title": "Config",
            "type": "object",
            "properties": {
                "retries": {"type": "integer", "default": 3},
                "name": {"type": "string", "default": "x"},
            },
        }
    )
    fields = own_fields(model)
    assert fields["retries"].default == 3
    assert fields["name"].default == "x"
    instance = model()
    assert (instance.retries, instance.name) == (3, "x")


# --------------------------------------------------------------------------- #
# nullable                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "nickname_schema",
    [
        {"type": ["string", "null"]},  # type-array nullable
        {"anyOf": [{"type": "string"}, {"type": "null"}]},  # anyOf nullable
    ],
)
def test_nullable(nickname_schema):
    model = convert(
        {"title": "N", "type": "object", "properties": {"nickname": nickname_schema}}
    )
    annotation = own_fields(model)["nickname"].annotation
    assert is_optional(annotation)
    assert core_type(annotation) is str


# --------------------------------------------------------------------------- #
# oneOf / anyOf / allOf                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_union_from_anyof_and_oneof(keyword):
    model = convert(
        {
            "title": "U",
            "type": "object",
            "properties": {"value": {keyword: [{"type": "string"}, {"type": "integer"}]}},
        }
    )
    annotation = own_fields(model)["value"].annotation
    assert set(non_none_args(annotation)) == {str, int}


def test_allof_single_ref_resolves():
    model = convert(
        {
            "title": "Employee",
            "type": "object",
            "properties": {"role": {"allOf": [{"$ref": "#/$defs/Role"}]}},
            "$defs": {"Role": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
    )
    role_type = core_type(own_fields(model)["role"].annotation)
    assert issubclass(role_type, DataPoint)


# --------------------------------------------------------------------------- #
# additionalProperties (must not break conversion)                            #
# --------------------------------------------------------------------------- #
def test_additional_properties_does_not_break_conversion():
    model = convert(
        {
            "title": "Bag",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "string"},
        }
    )
    assert "name" in own_fields(model)


# --------------------------------------------------------------------------- #
# class-name sanitization                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "title, expected",
    [
        ("Programming Language", "ProgrammingLanguage"),
        ("Programming-Language (v2)!", "ProgrammingLanguageV2"),
        ("myGraphModel", "MyGraphModel"),
        ("Person", "Person"),
        ("true", "TrueModel"),  # sanitizes to Python keyword "True"
    ],
)
def test_title_is_sanitized_to_valid_class_name(title, expected):
    model = convert({"title": title, "type": "object", "properties": {"name": {"type": "string"}}})
    assert model.__name__ == expected


def test_missing_title_uses_default_name():
    model = convert({"type": "object", "properties": {"name": {"type": "string"}}})
    assert model.__name__ == "GraphModel"
    assert issubclass(model, DataPoint)


def test_conversion_does_not_mutate_input_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    before = copy.deepcopy(schema)
    convert(schema)
    assert schema == before  # no 'title' injected into the caller's dict


# --------------------------------------------------------------------------- #
# round-trip: schema -> model -> schema                                        #
# --------------------------------------------------------------------------- #
def _array_items(prop: dict):
    """Return the ``items`` schema of an array property, seeing through Optional anyOf."""
    if prop.get("type") == "array":
        return prop.get("items")
    for branch in prop.get("anyOf", []):
        if branch.get("type") == "array":
            return branch.get("items")
    return None


def test_round_trip_preserves_structure():
    schema = {
        "title": "Book",
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "pages": {"type": "integer"},
            "author": {"$ref": "#/$defs/Author"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "author"],
        "$defs": {"Author": {"type": "object", "properties": {"name": {"type": "string"}}}},
    }

    model = convert(schema)
    round_tripped = graph_model_to_graph_schema(model)

    assert round_tripped["title"] == "Book"
    assert set(round_tripped["properties"]) == {"title", "pages", "author", "tags"}
    # required-ness survives; the optional field stays out of `required`
    assert set(round_tripped["required"]) == {"title", "author"}
    assert "pages" not in round_tripped.get("required", [])
    # nested object survives as a $ref into $defs
    assert "$ref" in round_tripped["properties"]["author"]
    assert "Author" in round_tripped["$defs"]
    # array-of-scalars survives
    assert _array_items(round_tripped["properties"]["tags"]) == {"type": "string"}


def test_round_trip_recursive_model():
    """A self-referential graph node (Person knows Persons) survives the round-trip."""

    class Person(DataPoint):
        name: str
        friends: "list[Person]" = []

    Person.model_rebuild()

    schema = graph_model_to_graph_schema(Person)
    model = convert(schema)

    assert issubclass(model, DataPoint)
    fields = own_fields(model)
    assert set(fields) == {"name", "friends"}

    friends_type = core_type(fields["friends"].annotation)
    assert get_origin(friends_type) is list
    item_type = get_args(friends_type)[0]
    assert issubclass(item_type, DataPoint)
