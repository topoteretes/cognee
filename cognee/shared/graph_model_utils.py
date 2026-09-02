import asyncio
import logging
import re
import sys
import types
from functools import lru_cache
from pprint import pprint
from typing import Any, Literal, Union, cast, get_args, get_origin

from datamodel_code_generator import DataModelType, GenerateConfig, InputFileType, generate
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticUndefined

import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.engine import DataPoint, Edge
from cognee.infrastructure.engine.models.FieldAnnotations import _FromIdentity
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.graph.utils.get_graph_from_model import collect_stored_data_points
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.graph.exceptions import InvalidReferenceTypeError

logger = logging.getLogger(__name__)


def _reference_target_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (list,):
        return _reference_target_type(get_args(annotation)[0])
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _reference_target_type(args[0])
    return annotation


def from_identity_fields(model: type[BaseModel]) -> dict[str, type[DataPoint]]:
    """Map field name to the DataPoint type a FromIdentity field refers to."""
    fields: dict[str, type[DataPoint]] = {}
    for name, field_info in model.model_fields.items():
        if any(isinstance(meta, _FromIdentity) for meta in field_info.metadata):
            fields[name] = _reference_target_type(field_info.annotation)
    return fields


def _single_identity_field(target_type: type[DataPoint], owning_field: str) -> str:
    identity = target_type._get_identity_fields()
    if not identity or len(identity) != 1:
        raise InvalidReferenceTypeError(
            f"{target_type.__name__} on {owning_field} needs exactly one identity_fields "
            f"entry, got {identity!r}"
        )
    return identity[0]


def _check_single_identity_field(target_type: type[DataPoint], owning_field: str) -> None:
    _single_identity_field(target_type, owning_field)


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


def _from_identity_llm_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (list,):
        return list[str]
    if origin in (Union, types.UnionType) and type(None) in get_args(annotation):
        return str | None
    return str


def _list_edge_inner(annotation: Any) -> type[Edge] | None:
    """Return the Edge[...] class if this is list[Edge[...]], else None."""
    if get_origin(annotation) not in (list,):
        return None
    inner = get_args(annotation)[0]
    if isinstance(inner, type) and issubclass(inner, Edge):
        return inner
    return None


def _edge_type_args(inner: type[Edge]) -> tuple[type[DataPoint], type[DataPoint], Any]:
    metadata = getattr(inner, "__pydantic_generic_metadata__", None)
    if not isinstance(metadata, dict):
        raise InvalidReferenceTypeError(f"{inner!r} is not a parametrized Edge")
    args = metadata.get("args")
    if not isinstance(args, (list, tuple)) or len(args) < 3:
        raise InvalidReferenceTypeError(f"{inner!r} is not a parametrized Edge")
    return args[0], args[1], args[2]


def edge_field_types(
    model: type[BaseModel],
) -> dict[str, tuple[type[DataPoint], type[DataPoint], Any]]:
    """Map each list[Edge[...]] field to (source type, target type, third Edge generic)."""
    fields: dict[str, tuple[type[DataPoint], type[DataPoint], Any]] = {}
    for name, field_info in model.model_fields.items():
        inner = _list_edge_inner(field_info.annotation)
        if inner is None:
            continue
        fields[name] = _edge_type_args(inner)
    return fields


def _row_class_name(field_name: str) -> str:
    return "".join(part.title() for part in field_name.split("_")) + "Edge"


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
    edge_types = edge_field_types(model)[field_name]
    source_type, target_type, _ = edge_types
    return source_type, target_type, _edge_row_model_for(field_name, edge_types)


def _llm_edge_field(
    model_type: type[BaseModel], field_name: str, field_info: Any, default_value: Any
) -> tuple | None:
    if _list_edge_inner(field_info.annotation) is None:
        return None
    *_, row_model = _edge_field_spec(model_type, field_name)
    return (
        types.GenericAlias(list, (row_model,)),
        default_value if default_value is not PydanticUndefined else PydanticUndefined,
    )


def _rewrite_from_identity(value: Any, target_type: type[DataPoint]) -> Any:
    identity_field = _single_identity_field(target_type, target_type.__name__)
    if value is None:
        return None
    if isinstance(value, list):
        return [_rewrite_from_identity(item, target_type) for item in value]
    if isinstance(value, str):
        return {identity_field: value}
    return value


def _llm_dump_to_model_dump(
    value: Any, model_class: type[BaseModel], path: tuple[Any, ...] = ()
) -> tuple[Any, dict]:
    if not isinstance(value, dict) or not (
        isinstance(model_class, type) and issubclass(model_class, BaseModel)
    ):
        return value, {}

    is_datapoint = issubclass(model_class, DataPoint)
    refs = from_identity_fields(model_class) if is_datapoint else {}
    edges = edge_field_types(model_class) if is_datapoint else {}
    converted: dict[str, Any] = {}
    rows: dict = {}
    for name, item in value.items():
        if name in edges:
            converted[name] = []
            if isinstance(item, list) and item:
                rows[(path, name)] = item
            continue
        if name in refs:
            converted[name] = _rewrite_from_identity(item, refs[name])
            continue
        field_info = model_class.model_fields.get(name)
        if field_info is None:
            converted[name] = item
            continue
        nested_dump, nested_rows = _descend_llm_dump(item, field_info.annotation, (*path, name))
        converted[name] = nested_dump
        rows.update(nested_rows)

    for name in edges:
        if name not in converted:
            converted[name] = []

    return converted, rows


def _descend_llm_dump(value: Any, annotation: Any, path: tuple[Any, ...]) -> tuple[Any, dict]:
    origin = get_origin(annotation)
    if origin in (list,):
        if not isinstance(value, list):
            return value, {}
        inner = get_args(annotation)[0]
        converted_items = []
        rows: dict = {}
        for index, item in enumerate(value):
            converted, nested = _descend_llm_dump(item, inner, (*path, index))
            converted_items.append(converted)
            rows.update(nested)
        return converted_items, rows
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _descend_llm_dump(value, args[0], path)
        return value, {}
    if isinstance(annotation, type) and issubclass(annotation, DataPoint):
        return _llm_dump_to_model_dump(value, annotation, path)
    return value, {}


def _instance_at(root: DataPoint, path: tuple[Any, ...]) -> Any:
    current: Any = root
    for step in path:
        current = current[step] if isinstance(step, int) else getattr(current, step)
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
        logger.warning("Skipping unresolved edge on %s: %s", field_name, row)
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
    ).normalize(source_node, field_name, target=target_node)


async def _attach_edge_rows(root: DataPoint, rows: dict) -> None:
    stored = await collect_stored_data_points(root)
    index = {(type(node), node.id): node for node in stored}
    for (path, field_name), row_dicts in rows.items():
        owner = _instance_at(root, path)
        source_type, target_type, row_model = _edge_field_spec(type(owner), field_name)
        built = []
        for row in row_dicts:
            edge = _resolve_edge_row(row, row_model, index, source_type, target_type, field_name)
            if edge is not None:
                built.append(edge)
        setattr(owner, field_name, built)


async def content_graph_to_data_point(content_graph: BaseModel, graph_model: type[DataPoint]):
    dump, rows = _llm_dump_to_model_dump(content_graph.model_dump(), graph_model)
    root = graph_model.model_validate(dump)
    if rows:
        await _attach_edge_rows(root, rows)
    return root


def datapoint_model_to_basemodel(
    model: type[BaseModel], *, strip_metadata: bool = False
) -> type[BaseModel]:
    """
    Convert a DataPoint-derived model into a plain BaseModel-derived model at runtime.

    Keeps domain fields from the model and its custom DataPoint parents. Drops fields
    defined on DataPoint itself (id, version, metadata, and other infrastructure).
    """

    def _replace_datapoint_types(
        annotation: Any, cache: dict[type[BaseModel], type[BaseModel]]
    ) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            if (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and issubclass(annotation, DataPoint)
            ):
                return _to_base_model(annotation, cache)
            return annotation

        if origin in (list, set, frozenset):
            inner = _replace_datapoint_types(args[0], cache)
            return origin[inner]

        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple[_replace_datapoint_types(args[0], cache), ...]  # ty:ignore[invalid-type-form]
            return tuple[tuple(_replace_datapoint_types(arg, cache) for arg in args)]  # ty:ignore[invalid-type-form]

        if origin is dict:
            key_type = _replace_datapoint_types(args[0], cache)
            value_type = _replace_datapoint_types(args[1], cache)
            return dict[key_type, value_type]

        if origin in (Union, types.UnionType):
            return Union[tuple(_replace_datapoint_types(arg, cache) for arg in args)]

        return annotation

    def _to_base_model(
        model_type: type[BaseModel], cache: dict[type[BaseModel], type[BaseModel]]
    ) -> type[BaseModel]:
        if model_type in cache:
            return cache[model_type]
        # Break potential cycles in nested model graphs (A -> B -> A).
        cache[model_type] = model_type

        class ConfiguredBase(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

        model_fields = model_type.model_fields

        # Keep merged domain fields; drop DataPoint infrastructure (including metadata).
        if issubclass(model_type, DataPoint):
            field_names = [name for name in model_fields if name not in DataPoint.model_fields]
        else:
            field_names = list(model_fields.keys())

        if strip_metadata:
            field_names = [name for name in field_names if name != "metadata"]

        converted_fields: dict[str, Any] = {}
        for field_name in field_names:
            field_info = model_fields[field_name]
            default_value = (
                Field(default_factory=field_info.default_factory)
                if field_info.default_factory is not None
                else field_info.default
            )
            if any(isinstance(meta, _FromIdentity) for meta in field_info.metadata):
                target_type = _reference_target_type(field_info.annotation)
                _check_single_identity_field(target_type, field_name)
                _check_constructible_from_identity(target_type, field_name)
                identity_field = target_type._get_identity_fields()[0]
                description = (
                    f"The {identity_field} of a {target_type.__name__} already in the graph"
                )
                if field_info.default_factory is not None:
                    field_default = Field(
                        default_factory=field_info.default_factory, description=description
                    )
                elif default_value is not PydanticUndefined:
                    field_default = Field(default=default_value, description=description)
                else:
                    field_default = Field(description=description)
                converted_fields[field_name] = (
                    _from_identity_llm_annotation(field_info.annotation),
                    field_default,
                )
                continue
            edge_field = _llm_edge_field(model_type, field_name, field_info, default_value)
            if edge_field is not None:
                converted_fields[field_name] = edge_field
                continue
            converted_fields[field_name] = (
                _replace_datapoint_types(field_info.annotation, cache),
                default_value if default_value is not PydanticUndefined else PydanticUndefined,
            )

        converted_model = create_model(
            model_type.__name__, __base__=ConfiguredBase, **converted_fields
        )
        converted_model.model_rebuild()
        cache[model_type] = converted_model

        return converted_model

    if not issubclass(model, DataPoint):
        return model

    return _to_base_model(model, {})


def graph_schema_to_graph_model(pydantic_json_schema: dict) -> BaseModel:
    # If a custom graph model is provided, convert it from dict to a Pydantic model class
    config = GenerateConfig(
        input_file_type=InputFileType.JsonSchema,
        input_filename="dynamic.json",
        output_model_type=DataModelType.PydanticV2BaseModel,
        additional_imports=["cognee.infrastructure.engine.DataPoint", "typing.Any", "typing"],
        # Set the base class for all generated models to the existing DataPoint class to
        # ensure proper integration with Cognee's graph engine
        base_class="cognee.infrastructure.engine.DataPoint",
        type_overrides={"DataPoint": "cognee.infrastructure.engine.DataPoint"},
    )
    # Override title to ensure a valid and secure Python class name for the generated model
    # 'config' has 'output=None', 'generate' is supposed to return a string
    result = cast(str, generate(pydantic_json_schema, config=config))

    # Replace the generated DataPointModel class definition made by datamodel_code_generator with
    # the existing Cognee DataPoint class
    # TODO: Probably not needed this was an attempt to allow DataPoint class to be inherited for input models
    result = re.sub(
        r"class DataPointModel\(DataPoint\):.*?(?=\nclass|\Z)", "", result, flags=re.DOTALL
    )
    # Replace all remaining references
    result = result.replace("DataPointModel", "DataPoint")

    # Dynamically create a module to execute the generated code and retrieve the model class
    # This is necessary to properly handle imports and references in the generated code
    module_name = "cognee.shared._generated_graph_models"
    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod

    exec(result, mod.__dict__)
    namespace = mod.__dict__

    # Extract the generated graph model class from the module's namespace
    graph_model = namespace[pydantic_json_schema["title"]]
    # Rebuild the DataPoint class first
    namespace["DataPoint"].model_rebuild()
    # Then rebuild the graph model to ensure it properly inherits from the updated DataPoint class
    graph_model.model_rebuild(_types_namespace=namespace)

    # Return dynamically created Pydantic model class that can be used in cognee for graph creation and querying
    return graph_model


def graph_model_to_graph_schema(graph_model: type[BaseModel]) -> dict:
    class GenerateJsonSchemaWithoutDefaultTitles(GenerateJsonSchema):
        def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
            return_value = super().field_title_should_be_set(schema)
            if return_value and is_core_schema(schema):
                return False
            return return_value

    model_for_schema = datapoint_model_to_basemodel(graph_model)
    return model_for_schema.model_json_schema(
        schema_generator=GenerateJsonSchemaWithoutDefaultTitles
    )


if __name__ == "__main__":

    async def main():
        # Create a clean slate for cognee -- reset data and system state
        print("Resetting cognee data...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        print("Data reset complete.\n")

        text = (
            "Python is an interpreted, high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991. "
            + "Python is widely used in data analysis, web development, and machine learning."
        )

        await cognee.add(text)

        # Define a custom graph model for programming languages.
        # Note: Models for generating graph schema can't inherit DataPoint directly, but will be set to inherit from
        # DataPoint in the graph_schema_to_model function later on
        class FieldType(BaseModel):
            name: str = "Field"
            metadata: dict = {"index_fields": ["name"]}

        class Field(BaseModel):
            name: str
            is_type: FieldType
            metadata: dict = {"index_fields": ["name"]}

        class ProgrammingLanguageType(BaseModel):
            name: str = "Programming Language"
            metadata: dict = {"index_fields": ["name"]}

        class ProgrammingLanguage(BaseModel):
            name: str
            used_in: list[Field] = []
            is_type: ProgrammingLanguageType
            metadata: dict = {"index_fields": ["name"]}

        # Transform the custom graph model to a JSON schema and then back to a Pydantic model class to ensure it is
        # properly formatted for cognee's graph engine
        graph_model_schema = graph_model_to_graph_schema(ProgrammingLanguage)

        graph_model = graph_schema_to_graph_model(graph_model_schema)

        # Use LLMs and cognee to create knowledge graph
        await cognee.cognify(graph_model=graph_model)

        query_text = "Tell me about Python and Rust"
        print(f"Searching cognee for insights with query: '{query_text}'")
        # Query cognee for insights on the added text
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
        )

        print("Search results:")
        # Display results
        for result_text in search_results:
            pprint(result_text)

        # Generate interactive graph visualization
        print("\nGenerating graph visualization...")
        from cognee.api.v1.visualize import visualize_graph

        await visualize_graph()
        print("Visualization saved to ~/graph_visualization.html")

    setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
