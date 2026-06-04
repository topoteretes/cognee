"""Pure builders for the HelixDB v2 dynamic-query JSON AST.

Every function returns a plain ``dict``/``list`` fragment that serializes to the
exact wire format the Helix gateway expects (``POST /v1/query``). Keeping the tag
encodings in one place lets them be unit-tested without a running server.

Encoding rules (serde, externally tagged unless noted):
  - unit variant   -> bare string: ``"Count"``
  - 1-field tuple  -> ``{"Var": <inner>}``
  - 2+-field tuple -> ``{"Var": [a, b]}``
  - struct variant -> ``{"Var": {"field": ...}}``
Literal values inside the AST use the tagged ``PropertyValue`` form
(``{"String": "x"}``); mutation/search inputs wrap those in ``PropertyInput``
(``{"Value": <PropertyValue>}``).
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from cognee.modules.storage.utils import JSONEncoder

# --------------------------------------------------------------------------- #
# PropertyValue (tagged literals)
# --------------------------------------------------------------------------- #


def property_value(value: Any) -> Dict[str, Any]:
    """Encode a scalar Python value as a tagged Helix ``PropertyValue``.

    Lists/dicts and other complex values are JSON-encoded to a string — graph
    properties are flattened the same way the Neptune adapter flattens them.
    """
    if isinstance(value, bool):
        return {"Bool": value}
    if isinstance(value, int):
        return {"I64": value}
    if isinstance(value, float):
        return {"F64": value}
    if isinstance(value, str):
        return {"String": value}
    if isinstance(value, UUID):
        return {"String": str(value)}
    if isinstance(value, datetime):
        return {"String": value.isoformat()}
    # Fallback: serialize structured/None values to a JSON string property.
    return {"String": json.dumps(value, cls=JSONEncoder)}


def f32_array(vector: Sequence[float]) -> Dict[str, Any]:
    """Encode a float vector as a ``F32Array`` PropertyValue."""
    return {"F32Array": [float(x) for x in vector]}


def string_array(values: Sequence[str]) -> Dict[str, Any]:
    return {"StringArray": [str(v) for v in values]}


def _input(value: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a ``PropertyValue`` as a literal ``PropertyInput`` (``{"Value": ...}``)."""
    return {"Value": value}


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #


def eq(prop: str, value: Any) -> Dict[str, Any]:
    """``Eq`` predicate — valid at both source (NWhere) and Where positions."""
    return {"Eq": [prop, property_value(value)]}


def and_(predicates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"And": predicates}


def or_(predicates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"Or": predicates}


def is_in(prop: str, values: Sequence[Any]) -> Dict[str, Any]:
    """``IsIn`` predicate (Where-only; not allowed in NWhere/EWhere)."""
    encoded = [property_value(v) for v in values]
    return {"IsIn": [prop, {"Array": encoded}]}


# --------------------------------------------------------------------------- #
# Source steps
# --------------------------------------------------------------------------- #


def n_where(predicate: Dict[str, Any]) -> Dict[str, Any]:
    return {"NWhere": predicate}


def n_by_var(var: str) -> Dict[str, Any]:
    return {"N": {"Var": var}}


def vector_search_nodes(
    label: str,
    prop: str,
    query_vector: Sequence[float],
    k: int,
    tenant_value: Optional[str] = None,
) -> Dict[str, Any]:
    """k-NN search over a node ``NodeVector`` index. Hits expose ``$distance``."""
    step: Dict[str, Any] = {
        "label": label,
        "property": prop,
        "query_vector": _input(f32_array(query_vector)),
        "k": {"Literal": int(k)},
    }
    step["tenant_value"] = (
        _input(property_value(tenant_value)) if tenant_value is not None else None
    )
    return {"VectorSearchNodes": step}


# --------------------------------------------------------------------------- #
# Traversal steps
# --------------------------------------------------------------------------- #


def out(label: Optional[str] = None) -> Dict[str, Any]:
    return {"Out": label}


def in_(label: Optional[str] = None) -> Dict[str, Any]:
    return {"In": label}


def both(label: Optional[str] = None) -> Dict[str, Any]:
    return {"Both": label}


# --------------------------------------------------------------------------- #
# Filters / shaping
# --------------------------------------------------------------------------- #


def where(predicate: Dict[str, Any]) -> Dict[str, Any]:
    return {"Where": predicate}


def limit(n: int) -> Dict[str, Any]:
    return {"Limit": int(n)}


def order_by(prop: str, descending: bool = False) -> Dict[str, Any]:
    return {"OrderBy": [prop, "Desc" if descending else "Asc"]}


DEDUP = "Dedup"
COUNT = "Count"


def project(entries: List[Dict[str, str]]) -> Dict[str, Any]:
    """``Project`` with untagged ``PropertyProjection`` entries (``source``/``alias``)."""
    return {"Project": entries}


def projection(source: str, alias: Optional[str] = None) -> Dict[str, str]:
    return {"source": source, "alias": alias if alias is not None else source}


def value_map(fields: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"ValueMap": fields}


# --------------------------------------------------------------------------- #
# Mutations
# --------------------------------------------------------------------------- #


def add_node(label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    """``AddN`` from a property dict (each value encoded as a literal PropertyInput)."""
    props = [[key, _input(property_value(val))] for key, val in properties.items()]
    return {"AddN": {"label": label, "properties": props}}


def add_node_with_inputs(label: str, properties: List[List[Any]]) -> Dict[str, Any]:
    """``AddN`` where property values are already encoded as ``PropertyInput`` dicts.

    Used for vector properties: ``[["Entity_name", {"Value": {"F32Array": [...]}}]]``.
    """
    return {"AddN": {"label": label, "properties": properties}}


def add_edge(label: str, to_var: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    """``AddE`` from the current node stream to the nodes stored in ``to_var``."""
    props = [[key, _input(property_value(val))] for key, val in properties.items()]
    return {"AddE": {"label": label, "to": {"Var": to_var}, "properties": props}}


def input_value(value: Any) -> Dict[str, Any]:
    """A literal ``PropertyInput`` wrapping an encoded scalar value."""
    return _input(property_value(value))


def input_vector(vector: Sequence[float]) -> Dict[str, Any]:
    """A literal ``PropertyInput`` wrapping an ``F32Array`` vector."""
    return _input(f32_array(vector))


def input_string_array(values: Sequence[str]) -> Dict[str, Any]:
    return _input(string_array(values))


def set_property(prop: str, property_input: Dict[str, Any]) -> Dict[str, Any]:
    """``SetProperty`` step (value already encoded as a ``PropertyInput``)."""
    return {"SetProperty": [prop, property_input]}


DROP = "Drop"


# --------------------------------------------------------------------------- #
# Indexes
# --------------------------------------------------------------------------- #


def create_node_equality_index(
    label: str, prop: str, unique: bool = False, if_not_exists: bool = True
) -> Dict[str, Any]:
    return {
        "CreateIndex": {
            "spec": {"NodeEquality": {"label": label, "property": prop, "unique": unique}},
            "if_not_exists": if_not_exists,
        }
    }


def create_node_vector_index(
    label: str,
    prop: str,
    tenant_property: Optional[str] = None,
    if_not_exists: bool = True,
) -> Dict[str, Any]:
    return {
        "CreateIndex": {
            "spec": {
                "NodeVector": {
                    "label": label,
                    "property": prop,
                    "tenant_property": tenant_property,
                }
            },
            "if_not_exists": if_not_exists,
        }
    }


# --------------------------------------------------------------------------- #
# Batch entries
# --------------------------------------------------------------------------- #


def query_entry(
    name: Optional[str],
    steps: List[Any],
    condition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"Query": {"name": name, "steps": steps, "condition": condition}}
