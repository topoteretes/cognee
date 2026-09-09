"""A custom graph model's LLM boundary: out to the LLM, and back.

``datapoint_model_to_basemodel`` goes out: it builds the model an LLM can answer, with
DataPoint infrastructure fields dropped, ``FromIdentity`` references narrowed to
identity strings, and ``list[Edge[...]]`` fields replaced by flat edge rows.
``content_graph_to_data_point`` comes back: identity strings become nested objects,
edge rows are resolved against the stored nodes, and a row that cannot be resolved is
dropped with a warning rather than sinking the chunk.

Naming, both ways round the trip: a function is named for what it produces.
``_llm_*_for`` goes out, ``_datapoint_*_for`` comes back. The two halves are
counterparts in effect, not mirrors in signature: going out transforms *types*
(class -> class, annotation -> annotation) and nothing is ever serialized; coming back
transforms *values* read from ``model_dump()``. Keep the levels paired — model with
dict, annotation with value — and whatever shape one half descends into, the other has
to as well, or LLM-answered rows arrive at ``model_validate`` raw and fail it. Every
recursion here starts by peeling ``Annotated`` with ``_strip_annotated``, so a marker
wrapper can never hide a shape from either half.
"""

import sys
import types
from functools import lru_cache
from typing import Annotated, Any, Literal, NamedTuple, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_core import PydanticUndefined

from cognee.infrastructure.engine import DataPoint, Edge
from cognee.infrastructure.engine.models.FieldAnnotations import _FromIdentity
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.graph.utils.get_graph_from_model import collect_stored_data_points
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.exceptions import InvalidReferenceTypeError

logger = get_logger()


# --- Annotation shapes -----------------------------------------------------------


def _strip_annotated(annotation: Any) -> tuple[Any, tuple]:
    """Peel every layer of ``Annotated`` off the top; return ``(core, markers)``."""
    markers: tuple = ()
    while get_origin(annotation) is Annotated:
        markers = (*markers, *annotation.__metadata__)
        annotation = annotation.__origin__
    return annotation, markers


def _is_union(origin: Any) -> bool:
    """True for both union spellings: ``Union[A, B]`` and ``A | B``."""
    return origin in (Union, types.UnionType)


def _non_none_members(args: tuple) -> list:
    """The union's members with ``None`` removed — what ``X | None`` says besides None."""
    return [arg for arg in args if arg is not type(None)]


def _annotation_markers(annotation: Any) -> list:
    """Every ``Annotated`` marker in an annotation, at any nesting depth."""
    core, markers = _strip_annotated(annotation)
    found = list(markers)
    for arg in get_args(core):
        if arg is type(None) or arg is Ellipsis:
            continue
        found.extend(_annotation_markers(arg))
    return found


def _field_markers(field_info: Any) -> list:
    """Every marker on a field, wherever in the annotation it sits.

    ``field_info.metadata`` only carries what an outermost ``Annotated`` held. The
    natural spelling for an optional reference puts it one level down —
    ``Annotated[Role, FromIdentity()] | None`` — and leaves the field's own metadata
    empty, so reading only ``field_info.metadata`` misses the marker entirely.
    """
    return [*field_info.metadata, *_annotation_markers(field_info.annotation)]


# --- FromIdentity references -----------------------------------------------------


def _identity_reference(
    annotation: Any, model: type[BaseModel], field_name: str
) -> tuple[type[DataPoint], Any]:
    """The DataPoint type a FromIdentity field refers to, and the LLM annotation for it.

    Supported spellings, each part optionally wrapped in ``Annotated``: ``Target``,
    ``Target | None``, ``list[Target]``, and ``list[Target] | None``. Anything else is
    rejected here, at model-build time: an unsupported shape would silently hand the
    LLM the whole nested class and mint a duplicate node instead of a reference.
    """
    core, _ = _strip_annotated(annotation)

    if isinstance(core, type) and issubclass(core, DataPoint):
        return core, str

    origin = get_origin(core)
    if origin is list:
        inner, _ = _strip_annotated(get_args(core)[0])
        if isinstance(inner, type) and issubclass(inner, DataPoint):
            return inner, list[str]
    if _is_union(origin):
        args = get_args(core)
        members = _non_none_members(args)
        if len(members) == 1 and len(args) == 2:
            target_type, llm_annotation = _identity_reference(members[0], model, field_name)
            return target_type, llm_annotation | None

    raise InvalidReferenceTypeError(
        f"{field_name} on {model.__name__} marks {annotation!r} with FromIdentity, but a "
        f"FromIdentity reference must name a DataPoint class as Target, Target | None, "
        f"or list[Target]. Any other shape would ask the LLM for a whole nested object "
        f"and mint a duplicate node instead of a reference."
    )


def _identity_field_types(model: type[BaseModel]) -> dict[str, type[DataPoint]]:
    """Map field name to the DataPoint type its FromIdentity marker refers to."""
    fields: dict[str, type[DataPoint]] = {}
    for name, field_info in model.model_fields.items():
        if any(isinstance(meta, _FromIdentity) for meta in _field_markers(field_info)):
            fields[name], _ = _identity_reference(field_info.annotation, model, name)
    return fields


def _single_identity_field(target_type: type[DataPoint], owning_field: str) -> str:
    identity = target_type._get_identity_fields()
    if not identity or len(identity) != 1:
        raise InvalidReferenceTypeError(
            f"{target_type.__name__} on {owning_field} needs exactly one identity_fields "
            f"entry, got {identity!r}"
        )
    return identity[0]


def _check_constructible_from_identity(target_type: type[DataPoint], owning_field: str) -> None:
    identity = set(target_type._get_identity_fields() or [])
    offending = [
        name
        for name, info in target_type.model_fields.items()
        if name not in DataPoint.model_fields and name not in identity and info.is_required()
    ]
    if offending:
        raise InvalidReferenceTypeError(
            f"{target_type.__name__} on {owning_field} has required fields that cannot "
            f"be filled from an identity string: {offending}"
        )


# --- Typed edge fields -----------------------------------------------------------


def _list_edge_inner(annotation: Any) -> type[Edge] | None:
    """Return the Edge[...] class if this is list[Edge[...]], optionally | None, else None."""
    core, _ = _strip_annotated(annotation)
    origin = get_origin(core)
    if _is_union(origin):
        members = _non_none_members(get_args(core))
        if len(members) == 1:
            return _list_edge_inner(members[0])
        return None
    if origin is not list:
        return None
    inner, _ = _strip_annotated(get_args(core)[0])
    if isinstance(inner, type) and issubclass(inner, Edge):
        return inner
    return None


def _contains_edge_annotation(annotation: Any) -> bool:
    """True when an ``Edge`` class appears anywhere in the annotation."""
    core, _ = _strip_annotated(annotation)
    if isinstance(core, type):
        return issubclass(core, Edge)
    return any(
        _contains_edge_annotation(arg)
        for arg in get_args(core)
        if arg is not type(None) and arg is not Ellipsis
    )


def _edge_endpoint_type(arg: Any, model: type[BaseModel], field_name: str) -> type[DataPoint]:
    """Resolve one ``Edge`` endpoint argument to the DataPoint class it names.

    ``Edge[...]`` records its arguments when it is subscripted, and nothing revisits
    them afterwards — not even ``model_rebuild(force=True)``. A same-type edge written
    on the node that owns it has to name its endpoints as strings, because the class is
    not bound inside its own body yet, so those strings are what arrives here. Resolve
    them against the owning model and its module, which is where such a reference
    points.
    """
    if isinstance(arg, type) and issubclass(arg, DataPoint):
        return arg

    name = getattr(arg, "__forward_arg__", arg)
    resolved: Any = None
    if isinstance(name, str):
        if name == model.__name__:
            resolved = model
        else:
            module = sys.modules.get(model.__module__)
            resolved = getattr(module, name, None) if module is not None else None
    if isinstance(resolved, type) and issubclass(resolved, DataPoint):
        return resolved

    raise InvalidReferenceTypeError(
        f"{field_name} on {model.__name__} names an Edge endpoint as {arg!r}, which is "
        f"not a DataPoint class. A typed Edge needs the endpoint classes themselves; a "
        f"string is only resolved against {model.__name__} and the module that defines "
        f"it. Declare the edge on a model defined after both endpoint classes — usually "
        f"the root graph model."
    )


def _edge_type_args(
    inner: type[Edge], model: type[BaseModel], field_name: str
) -> tuple[type[DataPoint], type[DataPoint], Any]:
    metadata = getattr(inner, "__pydantic_generic_metadata__", None)
    args = metadata.get("args") if isinstance(metadata, dict) else None
    if not isinstance(args, (list, tuple)) or len(args) < 3:
        raise InvalidReferenceTypeError(
            f"{field_name} on {model.__name__} declares a bare Edge. Parametrize it with "
            f"the endpoint classes, as in Edge[Source, Target]."
        )
    return (
        _edge_endpoint_type(args[0], model, field_name),
        _edge_endpoint_type(args[1], model, field_name),
        args[2],
    )


def _edge_field_types(
    model: type[BaseModel],
) -> dict[str, tuple[type[DataPoint], type[DataPoint], Any]]:
    """Map each list[Edge[...]] field to (source type, target type, third Edge generic).

    Endpoints come back as resolved classes, so ``_edge_row_model_for``'s cache is keyed
    on the classes rather than on whatever spelling the annotation used.
    """
    fields: dict[str, tuple[type[DataPoint], type[DataPoint], Any]] = {}
    for name, field_info in model.model_fields.items():
        inner = _list_edge_inner(field_info.annotation)
        if inner is None:
            continue
        fields[name] = _edge_type_args(inner, model, name)
    return fields


def _row_class_name(field_name: str) -> str:
    # Capitalize without lowercasing the rest: str.title() would collapse HTTP_link
    # and http_link into one class name, and pydantic's disambiguated fallback name
    # is what the LLM would then see.
    return "".join(part[:1].upper() + part[1:] for part in field_name.split("_")) + "Edge"


@lru_cache(maxsize=128)
def _edge_row_model_for(field_name: str, edge_types: tuple) -> type[BaseModel]:
    source_type, target_type, relationship_generic = edge_types
    source_id = _single_identity_field(source_type, field_name)
    target_id = _single_identity_field(target_type, field_name)
    fields: dict[str, Any] = {
        "source": (
            str,
            Field(description=f"The {source_id} of a {source_type.__name__} already in the graph"),
        ),
        "target": (
            str,
            Field(description=f"The {target_id} of a {target_type.__name__} already in the graph"),
        ),
    }
    if relationship_generic is str or get_origin(relationship_generic) is Literal:
        fields["relationship_type"] = (relationship_generic, ...)
    return create_model(_row_class_name(field_name), **fields)


def _edge_field_spec(
    model: type[BaseModel], field_name: str
) -> tuple[type[DataPoint], type[DataPoint], type[BaseModel]]:
    edge_types = _edge_field_types(model)[field_name]
    source_type, target_type, _ = edge_types
    return source_type, target_type, _edge_row_model_for(field_name, edge_types)


# --- Going out: the model the LLM answers ----------------------------------------


def datapoint_model_to_basemodel(
    model: type[BaseModel], *, strip_metadata: bool = False
) -> type[BaseModel]:
    """
    Convert a DataPoint-derived model into a plain BaseModel-derived model at runtime.

    Keeps domain fields from the model and its custom DataPoint parents. Drops fields
    defined on DataPoint itself (id, version, metadata, and other infrastructure).
    """
    if not issubclass(model, DataPoint):
        return model
    return _llm_model_for(model, {}, strip_metadata)


def _llm_model_for(
    model: type[BaseModel],
    cache: dict[type[BaseModel], type[BaseModel]],
    strip_metadata: bool,
) -> type[BaseModel]:
    """The class the LLM answers in ``model``'s place."""
    if model in cache:
        return cache[model]
    # Break potential cycles in nested model graphs (A -> B -> A).
    cache[model] = model

    class ConfiguredBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    model_fields = model.model_fields

    # Keep merged domain fields; drop DataPoint infrastructure (including metadata).
    if issubclass(model, DataPoint):
        field_names = [name for name in model_fields if name not in DataPoint.model_fields]
    else:
        field_names = list(model_fields.keys())

    if strip_metadata:
        field_names = [name for name in field_names if name != "metadata"]

    converted_fields = {
        field_name: _llm_field_for(
            model, field_name, model_fields[field_name], cache, strip_metadata
        )
        for field_name in field_names
    }

    converted_model = create_model(model.__name__, __base__=ConfiguredBase, **converted_fields)
    converted_model.model_rebuild()
    cache[model] = converted_model

    return converted_model


def _llm_field_for(
    model: type[BaseModel],
    field_name: str,
    field_info: Any,
    cache: dict,
    strip_metadata: bool,
) -> tuple[Any, Any]:
    """The ``(annotation, default)`` pair the LLM model declares for one field."""
    if any(isinstance(meta, _FromIdentity) for meta in _field_markers(field_info)):
        return _llm_identity_field_for(model, field_name, field_info)

    default_value = (
        Field(default_factory=field_info.default_factory)
        if field_info.default_factory is not None
        else field_info.default
    )

    edge_field = _llm_edge_field_for(model, field_name, field_info, default_value)
    if edge_field is not None:
        return edge_field

    if _contains_edge_annotation(field_info.annotation):
        raise InvalidReferenceTypeError(
            f"{field_name} on {model.__name__} declares Edge in a shape the LLM "
            f"extraction cannot fill. Supported: list[Edge[Source, Target]], optionally "
            f"| None. Anything else would hand the raw Edge schema to the LLM."
        )

    return (
        _llm_annotation_for(field_info.annotation, cache, strip_metadata),
        default_value if default_value is not PydanticUndefined else PydanticUndefined,
    )


def _llm_identity_field_for(
    model: type[BaseModel], field_name: str, field_info: Any
) -> tuple[Any, Any]:
    """A FromIdentity field, narrowed to the identity string(s) the LLM should answer."""
    target_type, llm_annotation = _identity_reference(field_info.annotation, model, field_name)
    identity_field = _single_identity_field(target_type, field_name)
    _check_constructible_from_identity(target_type, field_name)
    description = f"The {identity_field} of a {target_type.__name__} already in the graph"
    if field_info.default_factory is not None:
        field_default = Field(default_factory=field_info.default_factory, description=description)
    elif field_info.default is not PydanticUndefined:
        field_default = Field(default=field_info.default, description=description)
    else:
        field_default = Field(description=description)
    return llm_annotation, field_default


def _llm_edge_field_for(
    model: type[BaseModel], field_name: str, field_info: Any, default_value: Any
) -> tuple | None:
    """A list[Edge[...]] field, replaced by the flat rows the LLM should answer."""
    if _list_edge_inner(field_info.annotation) is None:
        return None
    *_, row_model = _edge_field_spec(model, field_name)
    row_list: Any = list[row_model]
    core, _ = _strip_annotated(field_info.annotation)
    if _is_union(get_origin(core)):
        # The declared field was optional; keep the LLM field optional too.
        row_list = row_list | None
    return (
        row_list,
        default_value if default_value is not PydanticUndefined else PydanticUndefined,
    )


def _llm_annotation_for(annotation: Any, cache: dict, strip_metadata: bool) -> Any:
    """The annotation the LLM answers in ``annotation``'s place.

    Rewrites every DataPoint class reachable through the annotation's shape; everything
    else passes through unchanged. Counterpart of ``_datapoint_value_for``.
    """
    core, markers = _strip_annotated(annotation)
    if markers:
        # Keep the markers: one of them may be a pydantic Field carrying a
        # description or a constraint the LLM schema should still show.
        return Annotated[(_llm_annotation_for(core, cache, strip_metadata), *markers)]

    origin = get_origin(core)
    args = get_args(core)

    if origin is None:
        if isinstance(core, type) and issubclass(core, DataPoint):
            return _llm_model_for(core, cache, strip_metadata)
        return core

    if origin in (list, set, frozenset):
        return origin[_llm_annotation_for(args[0], cache, strip_metadata)]

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple[_llm_annotation_for(args[0], cache, strip_metadata), ...]  # ty:ignore[invalid-type-form]
        return tuple[tuple(_llm_annotation_for(arg, cache, strip_metadata) for arg in args)]  # ty:ignore[invalid-type-form]

    if origin is dict:
        key_type = _llm_annotation_for(args[0], cache, strip_metadata)
        value_type = _llm_annotation_for(args[1], cache, strip_metadata)
        return dict[key_type, value_type]

    if _is_union(origin):
        return Union[tuple(_llm_annotation_for(arg, cache, strip_metadata) for arg in args)]

    return core


# --- Coming back: the answer into DataPoints -------------------------------------


class _MappingKey(NamedTuple):
    """A mapping key in a path, so it is not mistaken for an attribute name."""

    key: Any


async def content_graph_to_data_point(content_graph: BaseModel, graph_model: type[DataPoint]):
    """Validate an LLM answer as ``graph_model``, then resolve its typed edge rows."""
    dump, rows = _datapoint_dict_for(content_graph.model_dump(), graph_model)
    root = graph_model.model_validate(dump)
    if rows:
        await _attach_edge_rows(root, rows)
    return root


def _datapoint_dict_for(
    value: Any, model: type[BaseModel], path: tuple[Any, ...] = ()
) -> tuple[Any, dict]:
    """Build the dict ``model`` can validate from one LLM object.

    Counterpart of ``_llm_model_for``, which built the LLM class this object came from:
    that one strips a DataPoint down to what the LLM should answer, this one puts what
    the LLM answered back. Edge fields are emptied and their rows returned for
    ``_attach_edge_rows``; ``FromIdentity`` strings become nested objects.
    """
    if not isinstance(value, dict) or not (
        isinstance(model, type) and issubclass(model, BaseModel)
    ):
        return value, {}

    is_datapoint = issubclass(model, DataPoint)
    refs = _identity_field_types(model) if is_datapoint else {}
    edges = _edge_field_types(model) if is_datapoint else {}
    converted: dict[str, Any] = {}
    rows: dict = {}
    for name, item in value.items():
        if name in edges:
            converted[name] = []
            if isinstance(item, list) and item:
                rows[(path, name)] = item
            continue
        if name in refs:
            converted[name] = _datapoint_reference_for(item, refs[name])
            continue
        field_info = model.model_fields.get(name)
        if field_info is None:
            converted[name] = item
            continue
        nested_dump, nested_rows = _datapoint_value_for(item, field_info.annotation, (*path, name))
        converted[name] = nested_dump
        rows.update(nested_rows)

    for name in edges:
        if name not in converted:
            converted[name] = []

    return converted, rows


def _datapoint_value_for(value: Any, annotation: Any, path: tuple[Any, ...]) -> tuple[Any, dict]:
    """Build the model value for one LLM value. Counterpart of ``_llm_annotation_for``.

    These two must descend through every shape an LLM answer can actually arrive in, or
    the rows it holds go unread and reach ``model_validate`` as identity strings where a
    DataPoint is declared, which either raises or lets a looser union member swallow the
    node whole. ``set``/``frozenset`` are the one shape the forward pass rewrites that
    nothing can come back through: pydantic models are unhashable, so an answer holding
    a set of them fails validation before it ever reaches here.
    """
    core, _ = _strip_annotated(annotation)
    origin = get_origin(core)

    if origin is list:
        return _datapoint_sequence_for(value, get_args(core)[0], path)

    if origin is tuple:
        args = get_args(core)
        if len(args) == 2 and args[1] is Ellipsis:
            return _datapoint_sequence_for(value, args[0], path)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            return value, {}
        converted_items = []
        rows: dict = {}
        for index, (item, arg) in enumerate(zip(value, args)):
            converted, nested = _datapoint_value_for(item, arg, (*path, index))
            converted_items.append(converted)
            rows.update(nested)
        return converted_items, rows

    if origin is dict:
        if not isinstance(value, dict):
            return value, {}
        value_annotation = get_args(core)[1]
        converted_map: dict = {}
        rows = {}
        for key, item in value.items():
            converted, nested = _datapoint_value_for(
                item, value_annotation, (*path, _MappingKey(key))
            )
            converted_map[key] = converted
            rows.update(nested)
        return converted_map, rows

    if _is_union(origin):
        member = _union_member_for(value, get_args(core))
        if member is None:
            return value, {}
        return _datapoint_value_for(value, member, path)

    if isinstance(core, type) and issubclass(core, DataPoint):
        return _datapoint_dict_for(value, core, path)

    return value, {}


def _datapoint_sequence_for(value: Any, inner: Any, path: tuple[Any, ...]) -> tuple[Any, dict]:
    """Restore each item of a sequence, keeping the index it will validate into."""
    if not isinstance(value, (list, tuple)):
        return value, {}

    converted_items = []
    rows: dict = {}
    for index, item in enumerate(value):
        converted, nested = _datapoint_value_for(item, inner, (*path, index))
        converted_items.append(converted)
        rows.update(nested)
    return converted_items, rows


def _union_member_for(value: Any, args: tuple) -> Any:
    """Which union member the LLM likely answered as, or None when that is not decidable.

    The forward pass rewrote every DataPoint member of the union, so the LLM could have
    answered as any of them and a plain dump no longer records which. Score DataPoint
    members by how many of the value's keys they declare, first declared member winning
    ties. Pydantic's smart union decides by validating instead, so on an ambiguous
    answer the two can disagree; ``_attach_edge_rows`` re-checks the validated node
    before attaching rows, so a wrong guess degrades to a dropped row, not a crash.
    """
    members = _non_none_members(args)
    if len(members) == 1:
        return members[0]
    if not isinstance(value, dict):
        return None

    candidates = []
    for member in members:
        core, _ = _strip_annotated(member)
        if isinstance(core, type) and issubclass(core, DataPoint):
            candidates.append((member, core))
    if not candidates:
        return None

    best_member, best_core = max(
        candidates, key=lambda pair: len(value.keys() & pair[1].model_fields.keys())
    )
    if not value.keys() & best_core.model_fields.keys():
        return None
    return best_member


def _datapoint_reference_for(value: Any, target_type: type[DataPoint]) -> Any:
    """The nested-object value for one FromIdentity answer.

    Counterpart of ``_llm_identity_field_for``: that one narrowed the reference to an
    identity string, this one widens the answered string back to the dict the target
    class validates — item-wise through a list, untouched when it is neither.
    """
    identity_field = _single_identity_field(target_type, target_type.__name__)
    if value is None:
        return None
    if isinstance(value, list):
        return [_datapoint_reference_for(item, target_type) for item in value]
    if isinstance(value, str):
        return {identity_field: value}
    return value


def _instance_at(root: DataPoint, path: tuple[Any, ...]) -> Any:
    current: Any = root
    for step in path:
        if isinstance(step, _MappingKey):
            current = current[step.key]
        elif isinstance(step, int):
            current = current[step]
        else:
            current = getattr(current, step)
    return current


def _lookup_endpoint(index: dict, endpoint_type: type[DataPoint], identity_value: str):
    return index.get((endpoint_type, endpoint_type.id_for(identity_value)))


def _resolve_edge_row(
    row: dict,
    row_model: type[BaseModel],
    index: dict,
    source_type: type[DataPoint],
    target_type: type[DataPoint],
    field_name: str,
) -> Edge | None:
    validated = row_model.model_validate(row).model_dump()
    source_node = _lookup_endpoint(index, source_type, validated["source"])
    target_node = _lookup_endpoint(index, target_type, validated["target"])
    if source_node is None or target_node is None:
        logger.warning(
            "Skipping unresolved edge on %s: %s",
            field_name,
            row,
            extra={"field": field_name, "row": row, "edge_resolution_failed": True},
        )
        return None
    name = validated.get("relationship_type")
    rel_field = row_model.model_fields.get("relationship_type")
    if not isinstance(name, str):
        name = None
    elif rel_field is not None and rel_field.annotation is str:
        name = generate_edge_name(name)
    return Edge(
        source=source_node,
        target=target_node,
        relationship_type=name,
    ).fill_endpoints(source_node, field_name, target=target_node)


async def _attach_edge_rows(root: DataPoint, rows: dict) -> None:
    stored = await collect_stored_data_points(root)
    index = {(type(node), node.id): node for node in stored}
    for (path, field_name), row_dicts in rows.items():
        try:
            owner = _instance_at(root, path)
        except (AttributeError, KeyError, IndexError):
            logger.warning(
                "Dropping %d typed edge row(s) for %r: the validated graph has nothing at "
                "the position the LLM answered them under",
                len(row_dicts),
                field_name,
                extra={
                    "field": field_name,
                    "dropped_rows": len(row_dicts),
                    "edge_resolution_failed": True,
                },
            )
            continue
        if field_name not in _edge_field_types(type(owner)):
            logger.warning(
                "Dropping %d typed edge row(s) for %r: the validated node is a %s, which "
                "declares no such edge field",
                len(row_dicts),
                field_name,
                type(owner).__name__,
                extra={
                    "field": field_name,
                    "dropped_rows": len(row_dicts),
                    "edge_resolution_failed": True,
                },
            )
            continue
        source_type, target_type, row_model = _edge_field_spec(type(owner), field_name)
        built = []
        for row in row_dicts:
            edge = _resolve_edge_row(row, row_model, index, source_type, target_type, field_name)
            if edge is not None:
                built.append(edge)
        setattr(owner, field_name, built)
