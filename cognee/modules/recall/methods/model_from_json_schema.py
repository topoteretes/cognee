"""Rebuild a Pydantic model class from a client-supplied JSON Schema.

The recall HTTP API accepts structured-output requests as a JSON Schema
(``response_schema``, typically produced client-side via
``MyModel.model_json_schema()``) because a Python class cannot cross the HTTP
boundary. This module turns that schema back into a real ``BaseModel`` subclass
so the entire existing completion pipeline — instructor validation, retries,
result normalization — runs unchanged against it.

Deliberately supports only the structural subset Pydantic itself emits:
objects, primitives, arrays, enums, optionals/unions (``anyOf``), and nested
objects via ``$defs``/``definitions`` references. Anything outside that
(``allOf``, ``oneOf``, ``not``, recursive references) raises
``CogneeValidationError`` (HTTP 422) instead of half-validating. Value
constraints (``minLength``, ``minimum``, ...) are not enforced server-side —
clients re-validate against their own class when rehydrating ``structured``.
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, create_model

from cognee.exceptions import CogneeValidationError

_MAX_DEPTH = 10
_MAX_TOTAL_PROPERTIES = 200

_PRIMITIVES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_UNSUPPORTED_KEYWORDS = ("allOf", "oneOf", "not", "patternProperties")


def _fail(message: str) -> CogneeValidationError:
    return CogneeValidationError(message=f"response_schema: {message}")


class _Budget:
    """Caps total property count across the whole schema."""

    def __init__(self) -> None:
        self.properties = 0

    def spend(self, count: int) -> None:
        self.properties += count
        if self.properties > _MAX_TOTAL_PROPERTIES:
            raise _fail(f"more than {_MAX_TOTAL_PROPERTIES} properties in total")


def _resolve_ref(ref: str, defs: dict) -> tuple[str, dict]:
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            name = ref[len(prefix) :]
            if name not in defs:
                raise _fail(f"unresolvable reference '{ref}'")
            return name, defs[name]
    raise _fail(f"unsupported reference '{ref}' (only #/$defs/... and #/definitions/...)")


def _type_from(
    schema: Any,
    defs: dict,
    depth: int,
    in_flight_refs: frozenset[str],
    budget: _Budget,
) -> Any:
    """Translate one schema node into a Python annotation."""
    if depth > _MAX_DEPTH:
        raise _fail(f"nesting deeper than {_MAX_DEPTH} levels")
    if not isinstance(schema, dict):
        raise _fail("every schema node must be an object")

    for keyword in _UNSUPPORTED_KEYWORDS:
        if keyword in schema:
            raise _fail(f"unsupported JSON Schema keyword: {keyword}")

    if "$ref" in schema:
        name, target = _resolve_ref(schema["$ref"], defs)
        if name in in_flight_refs:
            raise _fail(f"recursive reference '{schema['$ref']}'")
        # A def may be any schema node (nested object, enum, ...), so recurse
        # through the general translator rather than assuming an object.
        return _type_from(target, defs, depth + 1, in_flight_refs | {name}, budget)

    if "enum" in schema:
        values = schema["enum"]
        if not values or not all(isinstance(v, (str, int, bool)) for v in values):
            raise _fail("enum values must be non-empty strings, integers, or booleans")
        return Literal[tuple(values)]

    if "anyOf" in schema:
        members = [
            _type_from(member, defs, depth + 1, in_flight_refs, budget)
            for member in schema["anyOf"]
        ]
        return Union[tuple(members)]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        members = [
            _type_from({**schema, "type": single}, defs, depth + 1, in_flight_refs, budget)
            for single in schema_type
        ]
        return Union[tuple(members)]

    if schema_type == "null":
        return type(None)
    if schema_type in _PRIMITIVES:
        return _PRIMITIVES[schema_type]
    if schema_type == "array":
        if "items" not in schema:
            raise _fail("array schemas must declare 'items'")
        return list[_type_from(schema["items"], defs, depth + 1, in_flight_refs, budget)]
    if schema_type == "object" or "properties" in schema:
        return _build_object(schema, defs, depth + 1, in_flight_refs, budget)

    raise _fail(f"unsupported or missing 'type' in schema node: {schema_type!r}")


def _build_object(
    schema: dict,
    defs: dict,
    depth: int,
    in_flight_refs: frozenset[str],
    budget: _Budget,
    fallback_name: str = "ResponseModel",
) -> type[BaseModel]:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        # An object with no declared properties carries no contract to enforce.
        raise _fail("object schemas must declare non-empty 'properties'")
    budget.spend(len(properties))

    required = set(schema.get("required", []))
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_schema in properties.items():
        if not field_name.isidentifier():
            raise _fail(f"property name '{field_name}' is not a valid identifier")
        annotation = _type_from(field_schema, defs, depth, in_flight_refs, budget)
        if field_name in required:
            fields[field_name] = (annotation, ...)
        else:
            fields[field_name] = (Optional[annotation], None)

    raw_name = schema.get("title") or fallback_name
    model_name = raw_name if raw_name.isidentifier() else fallback_name
    return create_model(model_name, **fields)


def model_from_json_schema(schema: dict) -> type[BaseModel]:
    """Build a ``BaseModel`` subclass from a JSON Schema dict.

    Raises ``CogneeValidationError`` (HTTP 422) for anything outside the
    supported structural subset — see module docstring.
    """
    if not isinstance(schema, dict):
        raise _fail("must be a JSON object")
    if schema.get("type") != "object":
        raise _fail("root schema must have type 'object'")

    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not isinstance(defs, dict):
        raise _fail("'$defs' must be an object")

    return _build_object(schema, defs, depth=0, in_flight_refs=frozenset(), budget=_Budget())
