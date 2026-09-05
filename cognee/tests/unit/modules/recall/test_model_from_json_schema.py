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

    def test_empty_any_of_rejected_as_schema_error(self):
        with pytest.raises(CogneeValidationError, match="anyOf.*at least one"):
            model_from_json_schema(
                {
                    "type": "object",
                    "properties": {"value": {"anyOf": []}},
                }
            )

    def test_empty_type_list_rejected_as_schema_error(self):
        with pytest.raises(CogneeValidationError, match="type.*at least one"):
            model_from_json_schema(
                {
                    "type": "object",
                    "properties": {"value": {"type": []}},
                }
            )

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
