from pydantic import create_model
from typing import Dict, Any, List, Optional, Union
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
            f"Could not map provided type: '{type_str}' to a supported type in Python."
        )


def dict_to_pydantic(schema_dict: Dict[str, Any], model_name: str = "DynamicModel"):
    """
    Convert a schema dictionary with string type annotations to a Pydantic model
    Handles nested dictionaries and lists
    """
    fields = {}

    for key, value in schema_dict.items():
        if isinstance(value, dict):
            # Check if it's a type annotation string or a nested schema
            if all(isinstance(v, str) for v in value.values()):
                # This is a simple type annotation dictionary
                pydantic_model = dict_to_pydantic(value, key)
                fields[key] = (pydantic_model, ...)
            else:
                # This is a nested schema - create a nested model
                nested_model_name = f"{model_name}_{key.capitalize()}"
                nested_model = dict_to_pydantic(value, nested_model_name)
                fields[key] = (nested_model, ...)
        elif isinstance(value, str):
            # Simple field with type annotation
            field_type = parse_type_string(value)
            fields[key] = (field_type, ...)
        else:
            raise TypeMappingKeyError(
                f"Value for key '{key}' must be a dictionary of type annotations or a nested schema."
            )

    # Create the model class
    DynamicModel = create_model(model_name, **fields)

    return DynamicModel


if __name__ == "__main__":
    # Example usage
    input_data = {
        "nodes": {"id": "str", "name": "str", "type": "str", "description": "str"},
        "edges": {"source_node_id": "str", "target_node_id": "str", "relationship_name": "str"},
    }

    # Create the dynamic model
    DynamicSchema = dict_to_pydantic(input_data, "KnowledgeGraphModel")

    # Test the dynamically generated model
    instance = DynamicSchema(
        nodes={
            "id": "node1",
            "name": "Test Node",
            "type": "test",
            "description": "This is a test node",
        },
        edges={
            "source_node_id": "node1",
            "target_node_id": "node2",
            "relationship_name": "connects_to",
        },
    )

    print(instance.model_dump())
    print(instance.model_json_schema())
