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
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        elif isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return str(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def copy_model(model: DataPoint, include_fields: dict = {}, exclude_fields: list = []):
    fields = {
        name: (field.annotation, field.default if field.default is not None else PydanticUndefined)
        for name, field in model.model_fields.items()
        if name not in exclude_fields
    }

    final_fields = {**fields, **include_fields}

    # Create a base class with the same configuration as DataPoint
    class ConfiguredBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    # Create the model inheriting from the configured base
    new_model = create_model(model.__name__, __base__=ConfiguredBase, **final_fields)

    new_model.model_rebuild()
    return new_model


def get_own_properties(data_point: DataPoint):
    properties = {}

    for field_name, field_value in data_point:
        if (
            field_name == "metadata"
            or isinstance(field_value, dict)
            or isinstance(field_value, DataPoint)
            or (isinstance(field_value, list) and isinstance(field_value[0], DataPoint))
        ):
            continue

        properties[field_name] = field_value

    return properties
