"""Which fields of a stored node have a vector-store collection.

A DataPoint's ``metadata`` names two kinds of stored fields: ``index_fields``
(embedded) and ``lexical_fields`` (kept in the keyword index, SDK-397). Each
has a collection ``<TypeName>_<field>`` whose row id is the node id; a field
in both shares one collection. Deleting a node must delete its rows in every
such collection, the lexical-only ones included: a cached document's text
lives only there.

The graph adapters snapshot a node's ``index_fields`` from its stored
properties when they plan a delete. The lexical fields are looked up from the
DataPoint class by type name instead, so every adapter, in tree or not, gets
them without a change.
"""

from collections.abc import Iterable


def vector_fields_of(metadata) -> list[str]:
    """The fields with a vector-store collection, from a node's ``metadata`` dict:
    ``index_fields`` first, then the ``lexical_fields`` not among them."""
    if not isinstance(metadata, dict):
        return []
    fields = list(metadata.get("index_fields") or [])
    for field in metadata.get("lexical_fields") or []:
        if field not in fields:
            fields.append(field)
    return fields


def lexical_fields_of_type(type_name: str | None) -> list[str]:
    """The ``lexical_fields`` the DataPoint subclass named ``type_name`` declares.

    Every class of that name counts (a name can be defined in two modules);
    an unknown name has none. Walks the subclass tree on each call: deletes
    ask once per node type, and a class defined after an earlier call is
    still found.
    """
    if not type_name:
        return []
    from cognee.infrastructure.engine.models.DataPoint import DataPoint

    found: list[str] = []
    stack = list(DataPoint.__subclasses__())
    seen: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if cls.__name__ != type_name:
            continue
        metadata_field = getattr(cls, "model_fields", {}).get("metadata")
        default = getattr(metadata_field, "default", None)
        for field in (default or {}).get("lexical_fields") or []:
            if field not in found:
                found.append(field)
    return found


def node_vector_fields(type_name: str | None, indexed_fields: Iterable[str] | None) -> list[str]:
    """``indexed_fields`` (a delete snapshot's embedded fields) plus the type's lexical fields."""
    fields = list(indexed_fields or [])
    for field in lexical_fields_of_type(type_name):
        if field not in fields:
            fields.append(field)
    return fields
