from enum import Enum
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, ValidationError

from cognee.exceptions import CogneeValidationError
from cognee.modules.recall.methods.model_from_json_schema import model_from_json_schema


class Ingredient(BaseModel):
    name: str
    grams: float


class Difficulty(str, Enum):
    easy = "easy"
    hard = "hard"


class Recipe(BaseModel):
    title: str
    servings: int
    vegetarian: bool
    difficulty: Difficulty
    ingredients: list[Ingredient]
    notes: Optional[str] = None
    kind: Literal["starter", "main", "dessert"] = "main"


class TestRoundtrip:
    """model -> model_json_schema() -> rebuilt model validates like the original."""

    def test_valid_payload_validates(self):
        rebuilt = model_from_json_schema(Recipe.model_json_schema())
        payload = {
            "title": "Stew",
            "servings": 4,
            "vegetarian": False,
            "difficulty": "easy",
            "ingredients": [{"name": "carrot", "grams": 200.0}],
            "kind": "main",
        }
        instance = rebuilt.model_validate(payload)
        assert Recipe.model_validate(
            instance.model_dump(exclude_none=True)
        ) == Recipe.model_validate(payload)

    def test_missing_required_field_rejected(self):
        rebuilt = model_from_json_schema(Recipe.model_json_schema())
        with pytest.raises(ValidationError):
            rebuilt.model_validate({"title": "Stew"})

    def test_wrong_nested_type_rejected(self):
        rebuilt = model_from_json_schema(Recipe.model_json_schema())
        with pytest.raises(ValidationError):
            rebuilt.model_validate(
                {
                    "title": "Stew",
                    "servings": 4,
                    "vegetarian": False,
                    "difficulty": "easy",
                    "ingredients": [{"name": "carrot", "grams": "many"}],
                }
            )

    def test_enum_value_outside_literal_rejected(self):
        rebuilt = model_from_json_schema(Recipe.model_json_schema())
        with pytest.raises(ValidationError):
            rebuilt.model_validate(
                {
                    "title": "Stew",
                    "servings": 4,
                    "vegetarian": False,
                    "difficulty": "impossible",
                    "ingredients": [],
                }
            )

    def test_optional_field_may_be_absent(self):
        rebuilt = model_from_json_schema(Recipe.model_json_schema())
        instance = rebuilt.model_validate(
            {
                "title": "Stew",
                "servings": 4,
                "vegetarian": False,
                "difficulty": "easy",
                "ingredients": [],
            }
        )
        assert instance.notes is None


class TestRejections:
    def test_non_object_root(self):
        with pytest.raises(CogneeValidationError):
            model_from_json_schema({"type": "string"})

    def test_unsupported_keyword(self):
        with pytest.raises(CogneeValidationError, match="allOf"):
            model_from_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"allOf": [{"type": "string"}]}},
                }
            )

    def test_recursive_ref(self):
        schema = {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/Node"}},
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                }
            },
        }
        with pytest.raises(CogneeValidationError, match="recursive"):
            model_from_json_schema(schema)

    def test_unresolvable_ref(self):
        with pytest.raises(CogneeValidationError, match="unresolvable"):
            model_from_json_schema(
                {"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}}
            )

    def test_empty_properties(self):
        with pytest.raises(CogneeValidationError, match="properties"):
            model_from_json_schema({"type": "object"})

    def test_array_without_items(self):
        with pytest.raises(CogneeValidationError, match="items"):
            model_from_json_schema({"type": "object", "properties": {"xs": {"type": "array"}}})

    def test_property_budget(self):
        schema = {
            "type": "object",
            "properties": {f"f{i}": {"type": "string"} for i in range(201)},
        }
        with pytest.raises(CogneeValidationError, match="properties in total"):
            model_from_json_schema(schema)

    def test_invalid_property_identifier(self):
        with pytest.raises(CogneeValidationError, match="identifier"):
            model_from_json_schema(
                {"type": "object", "properties": {"bad-name": {"type": "string"}}}
            )

    def test_empty_any_of(self):
        with pytest.raises(CogneeValidationError, match="anyOf"):
            model_from_json_schema(
                {
                    "type": "object",
                    "properties": {"value": {"anyOf": []}},
                    "required": ["value"],
                }
            )

    def test_non_list_any_of(self):
        with pytest.raises(CogneeValidationError, match="anyOf"):
            model_from_json_schema({"type": "object", "properties": {"value": {"anyOf": 5}}})

    def test_empty_type_list(self):
        with pytest.raises(CogneeValidationError, match="type"):
            model_from_json_schema({"type": "object", "properties": {"value": {"type": []}}})

    def test_nested_empty_any_of(self):
        with pytest.raises(CogneeValidationError, match="anyOf"):
            model_from_json_schema(
                {
                    "type": "object",
                    "properties": {"xs": {"type": "array", "items": {"anyOf": []}}},
                }
            )

    @pytest.mark.parametrize(
        "field_schema, match",
        [
            ({"enum": 5}, "enum"),
            ({"enum": "abc"}, "enum"),
            ({"type": {}}, "type"),
            ({"type": [{}]}, "type"),
            ({"$ref": 5}, r"\$ref"),
        ],
    )
    def test_malformed_keyword_value(self, field_schema, match):
        with pytest.raises(CogneeValidationError, match=match):
            model_from_json_schema({"type": "object", "properties": {"value": field_schema}})

    @pytest.mark.parametrize("required", [5, "value", [5]])
    def test_malformed_required(self, required):
        with pytest.raises(CogneeValidationError, match="required"):
            model_from_json_schema(
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": required,
                }
            )

    def test_non_string_title_falls_back_to_default_name(self):
        rebuilt = model_from_json_schema(
            {"type": "object", "title": 5, "properties": {"value": {"type": "string"}}}
        )
        assert rebuilt.__name__ == "ResponseModel"


class TestUnions:
    def test_single_member_any_of_collapses_to_that_type(self):
        rebuilt = model_from_json_schema(
            {
                "type": "object",
                "properties": {"value": {"anyOf": [{"type": "string"}]}},
                "required": ["value"],
            }
        )
        assert rebuilt.model_validate({"value": "ok"}).value == "ok"

    def test_type_list_builds_a_union(self):
        rebuilt = model_from_json_schema(
            {
                "type": "object",
                "properties": {"value": {"type": ["string", "null"]}},
                "required": ["value"],
            }
        )
        assert rebuilt.model_validate({"value": "ok"}).value == "ok"
        assert rebuilt.model_validate({"value": None}).value is None
