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
    """
    Create a new model instance from an existing DataPoint, optionally
    filtering fields via include_fields or exclude_fields.

    Args:
        model: The source DataPoint instance to copy.
        include_fields: If provided, only these fields are included in the copy.
        exclude_fields: Fields to exclude from the copy. Ignored if also in include_fields.

    Returns:
        A new model instance with the selected fields populated.
    """
    if exclude_fields is None:
        exclude_fields = []

    all_field_names = list(type(model).model_fields.keys())
    include_filter = set(
        include_fields) if include_fields is not None else None
    exclude_filter = set(exclude_fields)

    fields_to_copy = [
        name for name in all_field_names
        # explicitly included = always kept
        if (include_filter is not None and name in include_filter)
        # no include filter = exclude applies
        or (include_filter is None and name not in exclude_filter)
    ]

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

    instance_data = {
        name: getattr(model, name)
        for name in fields_to_copy
        if hasattr(model, name)
    }

    return new_model_class.model_construct(**instance_data)


def get_own_properties(data_point: DataPoint):
    """
    Extract non-metadata fields from a DataPoint, omitting nested DataPoint
    objects, dicts, and lists whose first element is a DataPoint instance.
    Primitive containers such as lists of non-DataPoint values may still be included.

    Args:
        data_point: The DataPoint instance to extract properties from.

    Returns:
        A dict of field names to their non-DataPoint field values.
    """
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
