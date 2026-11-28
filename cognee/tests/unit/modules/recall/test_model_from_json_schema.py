from enum import Enum
from typing import Literal

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
    notes: str | None = None
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
        with pytest.raises(CogneeValidationError, match="anyOf.*non-empty") as exc_info:
            model_from_json_schema({"type": "object", "properties": {"x": {"anyOf": []}}})
        assert exc_info.value.status_code == 422

    @pytest.mark.parametrize("any_of", [{}, {"type": "string"}, "string"])
    def test_non_list_any_of(self, any_of):
        with pytest.raises(CogneeValidationError, match="anyOf.*non-empty"):
            model_from_json_schema({"type": "object", "properties": {"x": {"anyOf": any_of}}})

    def test_empty_type_array(self):
        with pytest.raises(CogneeValidationError, match="type.*non-empty") as exc_info:
            model_from_json_schema({"type": "object", "properties": {"x": {"type": []}}})
        assert exc_info.value.status_code == 422


class TestUnions:
    """Non-empty ``anyOf`` and ``type`` arrays build a Union of their members."""

    def test_any_of_accepts_each_member(self):
        rebuilt = model_from_json_schema(
            {
                "type": "object",
                "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                "required": ["x"],
            }
        )
        assert rebuilt.model_validate({"x": "a"}).x == "a"
        assert rebuilt.model_validate({"x": 1}).x == 1
        with pytest.raises(ValidationError):
            rebuilt.model_validate({"x": [1]})

    def test_type_array_accepts_each_member(self):
        rebuilt = model_from_json_schema(
            {
                "type": "object",
                "properties": {"x": {"type": ["string", "null"]}},
                "required": ["x"],
            }
        )
        assert rebuilt.model_validate({"x": "a"}).x == "a"
        assert rebuilt.model_validate({"x": None}).x is None
        with pytest.raises(ValidationError):
            rebuilt.model_validate({"x": [1]})
