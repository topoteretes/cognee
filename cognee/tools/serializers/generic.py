"""Serialize Cognee tools to generic JSON Schema format.

Works with any framework that accepts JSON Schema tool definitions.
"""

import inspect
from typing import get_type_hints

from ..definitions import TOOLS


def for_generic() -> list:
    """Convert Cognee memory tools to generic JSON Schema format.

    Returns
    -------
    list
        A list of dicts with name, description, and parameters (JSON Schema).
    """
    result = []
    for fn in TOOLS:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            hint = hints.get(name, str)
            json_type = _python_type_to_json(hint)
            prop = {"type": json_type, "description": name}
            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(name)
            properties[name] = prop

        result.append({
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return result


def _python_type_to_json(hint) -> str:
    """Map a Python type hint to a JSON Schema type string."""
    if hint is int:
        return "integer"
    if hint is float:
        return "number"
    if hint is bool:
        return "boolean"
    return "string"
