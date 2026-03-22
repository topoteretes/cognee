import json
from uuid import UUID
from decimal import Decimal
from datetime import datetime
from pydantic_core import PydanticUndefined
from pydantic import create_model, ConfigDict, BaseModel

from cognee.infrastructure.engine import DataPoint


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def copy_model(
    model: DataPoint, include_fields: list | None = None, exclude_fields: list | None = None
):
    if include_fields is None:
        include_fields = []
    if exclude_fields is None:
        exclude_fields = []

    all_field_names = list(type(model).model_fields.keys())

    # Determine which fields to keep
    fields_to_copy = [
        name for name in all_field_names
        if name not in exclude_fields
        and (not include_fields or name in include_fields)
    ]

    # Build field definitions for create_model
    fields = {
        name: (
            type(model).model_fields[name].annotation,
            type(model).model_fields[name].default
            if type(model).model_fields[name].default is not None
            else PydanticUndefined
        )
        for name in fields_to_copy
    }

    class ConfiguredBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    new_model_class = create_model(
        type(model).__name__, __base__=ConfiguredBase, **fields)
    new_model_class.model_rebuild()

    # Use model's actual object values (not serialized dicts) to preserve nested models
    instance_data = {
        name: getattr(model, name)
        for name in fields_to_copy
        if hasattr(model, name)
    }

    # skip re-validation
    return new_model_class.model_construct(**instance_data)


def get_own_properties(data_point: DataPoint):
    properties = {}

    for field_name, field_value in data_point:
        if (
            field_name == "metadata"
            or isinstance(field_value, dict)
            or isinstance(field_value, DataPoint)
            or (
                isinstance(field_value, list)
                and len(field_value) > 0
                and isinstance(field_value[0], DataPoint)
            )
        ):
            continue

        properties[field_name] = field_value

    return properties
