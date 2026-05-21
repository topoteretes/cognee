from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint
from cognee.shared.graph_model_utils import graph_model_to_graph_schema


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
