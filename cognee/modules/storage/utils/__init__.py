import json
from uuid import UUID
from datetime import datetime
from pydantic_core import PydanticUndefined

from cognee.infrastructure.engine import DataPoint

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        elif isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return str(obj)
        return json.JSONEncoder.default(self, obj)


from pydantic import create_model

def copy_model(model: DataPoint, include_fields: dict = {}, exclude_fields: list = []):
    fields = {
        name: (field.annotation, field.default if field.default is not None else PydanticUndefined)
            for name, field in model.model_fields.items()
            if name not in exclude_fields
    }

    final_fields = {
        **fields,
        **include_fields
    }

    return create_model(model.__name__, **final_fields)

def get_own_properties(data_point: DataPoint):
    properties = {}

    for field_name, field_value in data_point:
        if field_name == "_metadata" \
            or isinstance(field_value, dict) \
            or isinstance(field_value, DataPoint) \
            or (isinstance(field_value, list) and isinstance(field_value[0], DataPoint)):
            continue

        properties[field_name] = field_value

    return properties
