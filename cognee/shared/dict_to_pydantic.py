from pydantic import create_model
from typing import Dict, Any, List, Optional
from cognee.shared.exceptions.exceptions import TypeMappingKeyError
import typing


def parse_type_string(type_str: str) -> type:
    """
    Convert a string type annotation to an actual Python type
    """
    type_str = type_str.lower()  # convert to lowercase for uniformity

    # Handle List[...] types
    if type_str.startswith("list[") and type_str.endswith("]"):
        inner_type_str = type_str[5:-1]  # Extract what's inside List[]
        inner_type = parse_type_string(inner_type_str)
        return List[inner_type]  # type: ignore

    # Handle Dict[...] types
    elif type_str.startswith("dict[") and type_str.endswith("]"):
        inner_types = type_str[5:-1].split(", ")
        key_type = parse_type_string(inner_types[0])
        value_type = parse_type_string(inner_types[1])
        return Dict[key_type, value_type]  # type: ignore

    # Handle Optional[...] types
    elif type_str.startswith("optional[") and type_str.endswith("]"):
        inner_type = parse_type_string(type_str[9:-1])
        return Optional[inner_type]  # type: ignore

    # Handle Union[...] types
    elif type_str.startswith("union[") and type_str.endswith("]"):
        inner_types = [parse_type_string(t.strip()) for t in type_str[6:-1].split(", ")]
        return typing.Union[tuple(inner_types)]  # type: ignore

    # Handle primitive types
    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "any": Any,
        "none": type(None),
    }

    try:
        return type_mapping[type_str]
    except KeyError:
        raise TypeMappingKeyError(
            f"Could not map provided type: {type_str} to a supported type in Python."
        )


def dict_to_pydantic(schema_dict: Dict[str, str], model_name: str = "DynamicModel"):
    """
    Convert a schema dictionary with string type annotations to a Pydantic model
    """
    fields = {}
    for key, type_str in schema_dict.items():
        # Parse the string type annotation to an actual Python type
        field_type = parse_type_string(type_str)
        fields[key] = (field_type, ...)  # ... means required field and no default value

    # Create the model class
    DynamicModel = create_model(model_name, **fields)

    # Return the model class (not an instance)
    return DynamicModel


if __name__ == "__main__":
    schema_dict = {"name": "str", "date": "str", "participants": "List[List[str]]"}

    # Create the model class
    CalendarEvent = dict_to_pydantic(schema_dict, "CalendarEvent")

    # Try validating some data against the model
    test_data = {
        "name": "Meeting",
        "date": "2024-01-01",
        "participants": [["Alice", "Bob"], ["Charlie", "Dave"]],
    }

    # Create an instance with actual data
    event = CalendarEvent(**test_data)
    print(event)
    print(f"Type of participants: {type(event.participants)}")
    print(f"First participant: {event.participants[0][0]}")
