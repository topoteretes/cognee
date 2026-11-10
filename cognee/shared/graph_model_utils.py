"""Custom graph models as JSON schema: a model to a schema, and a schema to a model.

This is the serialization boundary for custom graph models — what the HTTP API moves
across the wire. The LLM boundary (what an LLM is asked to answer for a model,
and how the answer becomes DataPoints again) lives in ``cognee.shared.llm_graph_model``.

The schema is the prompt model's: DataPoint infrastructure dropped, a ``FromIdentity``
reference narrowed to its identity string, a ``list[Edge[...]]`` field flattened to
rows. What plain JSON Schema cannot say rides on extension keys, so the receiving side
rebuilds the declaring class rather than the flattened one:

- ``x-cognee-edge`` on a row-list property names the endpoint classes,
  ``{"source": "Person", "target": "Person"}``. The relationship rule is the row's own
  ``relationship_type`` property: an ``enum``, a free string, or absent for the field name.
- ``x-cognee-from-identity`` on a reference property names the target class.

Both name a class in ``$defs``, so a class the prompt model only ever names as a string
is written there as well. Each class's ``metadata`` default (index and identity fields)
is carried the same way, as the spec compiler does. A consumer that ignores the
extension keys still reads a valid schema of the flat shapes.
"""

import re
import sys
import types
from typing import Annotated, Any, Literal, cast

from datamodel_code_generator import DataModelType, GenerateConfig, InputFileType, generate
from pydantic import BaseModel
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema, models_json_schema

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.engine import DataPoint, Edge, FromIdentity
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.exceptions import (
    ExternalSchemaReferenceError,
    InvalidGraphSchemaError,
    UnresolvedSchemaReferenceError,
)
from cognee.shared.llm_graph_model import (
    _datapoint_types_in,
    _edge_field_types,
    _identity_field_types,
    datapoint_models_to_basemodels,
    rebuild_models,
)

EDGE_KEY = "x-cognee-edge"
FROM_IDENTITY_KEY = "x-cognee-from-identity"


def _reject_external_refs(node) -> None:
    """Raise if any ``$ref`` in the schema points outside the document.

    The schema arrives over the HTTP API from the caller. ``datamodel-code-generator``
    resolves ``$ref`` values natively: an ``http(s)://`` URL is fetched from the server
    and a path is read from the server's disk. Only pointers into the same document
    (``#/$defs/Node``), which is what Pydantic emits for nested models, are legitimate.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            raise ExternalSchemaReferenceError(
                f"graph_model contains an external $ref ({ref!r}). Only in-document "
                "references starting with '#' are allowed; URLs and file paths are not fetched."
            )
        for value in node.values():
            _reject_external_refs(value)
    elif isinstance(node, list):
        for value in node:
            _reject_external_refs(value)


def resolve_graph_model(graph_model: type[BaseModel] | dict[str, Any] | None) -> type[BaseModel]:
    """The model class a cognify run extracts into, from either form a caller may pass.

    ``None`` is the default ``KnowledgeGraph``; a class is returned as it is; a dict is a
    JSON schema (what ``POST /cognify`` and ``POST /remember`` take, and what
    ``graph_model_to_graph_schema`` produces) and is built into a class here. This is the
    one place a schema becomes a class on the way into a run, so every schema passes the
    ``$ref`` guards of ``graph_schema_to_graph_model``. Any schema the converter cannot
    build raises a ``CogneeValidationError`` (HTTP 400), so a caller can tell a bad
    schema from every other failure.
    """
    if graph_model is None:
        return KnowledgeGraph
    if isinstance(graph_model, type):
        return graph_model
    if not isinstance(graph_model, dict):
        raise InvalidGraphSchemaError(
            "graph_model must be a model class or a JSON schema object, "
            f"got {type(graph_model).__name__}."
        )
    if not isinstance(graph_model.get("title"), str):
        raise InvalidGraphSchemaError(
            "graph_model schema needs a top-level 'title' naming its root class."
        )
    try:
        return graph_schema_to_graph_model(graph_model)
    except CogneeValidationError:
        raise
    except Exception as error:
        # The generator's own errors are untyped; they all mean the schema is unusable.
        raise InvalidGraphSchemaError(
            f"graph_model could not be built from the schema: {error}"
        ) from error


def graph_schema_to_graph_model(pydantic_json_schema: dict) -> BaseModel:
    _reject_external_refs(pydantic_json_schema)
    # If a custom graph model is provided, convert it from dict to a Pydantic model class
    config = GenerateConfig(
        # Second layer behind _reject_external_refs: the generator must never fetch
        # URLs or read files outside the schema it was given.
        allow_remote_refs=False,
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

    exec(result, mod.__dict__)  # noqa: S102 - runs the generated model module source on purpose
    namespace = mod.__dict__

    # Extract the generated graph model class from the module's namespace
    graph_model = namespace[pydantic_json_schema["title"]]
    # Rebuild the DataPoint class first
    namespace["DataPoint"].model_rebuild()
    # Then rebuild the graph model to ensure it properly inherits from the updated DataPoint class
    graph_model.model_rebuild(_types_namespace=namespace)

    _restore_typed_fields(pydantic_json_schema, namespace)

    # Return dynamically created Pydantic model class that can be used in cognee for graph creation and querying
    return graph_model


def _restore_typed_fields(schema: dict, namespace: dict) -> None:
    """Put ``Edge[...]`` and ``FromIdentity`` back on the regenerated classes.

    The generator built each extension-marked property as the flat type the schema
    shows: a row class, a string. Swapping the annotation on the generated ``FieldInfo``
    and rebuilding gives ``llm_graph_model`` the declaring shape it reads locally.
    """
    defs = schema.get("$defs", {})
    retyped = False
    for class_name, entry in {schema["title"]: schema, **defs}.items():
        for field_name, prop in entry.get("properties", {}).items():
            if EDGE_KEY in prop:
                annotation = _edge_annotation(prop, defs, namespace)
            elif FROM_IDENTITY_KEY in prop:
                annotation = _identity_annotation(prop, namespace)
            else:
                continue
            _generated_class(namespace, class_name).model_fields[field_name].annotation = annotation
            retyped = True
    if retyped:
        rebuild_models(
            [
                value
                for value in namespace.values()
                if isinstance(value, type)
                and issubclass(value, DataPoint)
                and value is not DataPoint
            ]
        )


def _edge_annotation(prop: dict, defs: dict, namespace: dict) -> Any:
    """``list[Edge[Source, Target, Rel]]`` for a row-list property, ``| None`` when nullable.

    ``Rel`` is read off the row: an ``enum`` becomes the ``Literal`` the LLM picks
    from, a plain string is free-form, and no ``relationship_type`` at all means the
    edge is named after the field.
    """
    endpoints = prop[EDGE_KEY]
    if not isinstance(endpoints, dict) or not {"source", "target"} <= endpoints.keys():
        raise UnresolvedSchemaReferenceError(
            f"{EDGE_KEY} must name a 'source' and a 'target' class, got {endpoints!r}."
        )
    source = _generated_class(namespace, endpoints["source"])
    target = _generated_class(namespace, endpoints["target"])
    core, nullable = _without_null(prop)
    items = core.get("items", {})
    row = defs.get(items["$ref"].rsplit("/", 1)[-1], {}) if "$ref" in items else items
    rule = row.get("properties", {}).get("relationship_type")
    if rule is None:
        edge = Edge[source, target]  # ty: ignore[invalid-type-form] - runtime-built generic
    elif "enum" in rule:
        edge = Edge[source, target, Literal[tuple(rule["enum"])]]  # ty: ignore[invalid-type-form] - runtime-built generic
    else:
        edge = Edge[source, target, str]  # ty: ignore[invalid-type-form] - runtime-built generic
    annotation: Any = list[edge]
    return annotation | None if nullable else annotation


def _identity_annotation(prop: dict, namespace: dict) -> Any:
    """``Annotated[Target, FromIdentity()]`` for a reference property, in the spelling
    the property's shape shows: a list for an array, ``| None`` when nullable."""
    target = _generated_class(namespace, prop[FROM_IDENTITY_KEY])
    core, nullable = _without_null(prop)
    reference: Any = list[target] if core.get("type") == "array" else target  # ty: ignore[invalid-type-form] - runtime-built alias
    annotation: Any = Annotated[reference, FromIdentity()]
    return annotation | None if nullable else annotation


def _without_null(prop: dict) -> tuple[dict, bool]:
    """The property without its ``null`` alternative, and whether it had one."""
    members = prop.get("anyOf")
    if not members:
        return prop, False
    non_null = [member for member in members if member != {"type": "null"}]
    if len(non_null) == 1 and len(non_null) < len(members):
        return non_null[0], True
    return prop, False


def _generated_class(namespace: dict, name: Any) -> type[DataPoint]:
    """The class the generator made for ``name``, which must be a ``$defs`` entry."""
    cls = namespace.get(name) if isinstance(name, str) else None
    if isinstance(cls, type) and issubclass(cls, DataPoint) and cls is not DataPoint:
        return cls
    raise UnresolvedSchemaReferenceError(
        f"graph_model names {name!r} as an edge endpoint or reference target, but its "
        f"$defs define no such class."
    )


def _datapoint_types_in_tree(graph_model: type[BaseModel]) -> dict[str, type[DataPoint]]:
    """Every DataPoint class in ``graph_model`` and its fields, by class name: the
    names the schema's ``$defs`` use. Edge endpoints and FromIdentity targets count,
    nested anywhere else or not."""
    found: dict[str, type[DataPoint]] = {}
    seen: set[type] = set()

    def walk(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.add(model)
        if issubclass(model, DataPoint):
            found[model.__name__] = model
        for name, field_info in model.model_fields.items():
            if name in DataPoint.model_fields:
                continue
            for nested in _datapoint_types_in(field_info.annotation):
                walk(nested)
        if issubclass(model, DataPoint):
            for source, target, _ in _edge_field_types(model).values():
                walk(source)
                walk(target)

    walk(graph_model)
    return found


def graph_model_to_graph_schema(graph_model: type[BaseModel]) -> dict:
    class GenerateJsonSchemaWithoutDefaultTitles(GenerateJsonSchema):
        def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
            return_value = super().field_title_should_be_set(schema)
            if return_value and is_core_schema(schema):
                return False
            return return_value

    classes = _datapoint_types_in_tree(graph_model)
    others = [cls for cls in classes.values() if cls is not graph_model]
    prompt_models = datapoint_models_to_basemodels([graph_model, *others])
    # One joint schema, so a class the root's prompt model never nests — a FromIdentity
    # target it names as a string, an edge endpoint — still gets its $defs entry.
    _, joint = models_json_schema(
        [(model, "validation") for model in prompt_models],
        ref_template="#/$defs/{model}",
        schema_generator=GenerateJsonSchemaWithoutDefaultTitles,
    )
    defs = joint["$defs"]
    schema = defs.pop(graph_model.__name__)
    if defs:
        schema["$defs"] = defs
    for name, model in classes.items():
        _stamp_declarations(schema if model is graph_model else defs[name], model)
    return schema


def _stamp_declarations(entry: dict, model: type[DataPoint]) -> None:
    """What the prompt model lost and the receiving side must get back: the class's
    ``metadata`` default, and the endpoints or target of each typed field."""
    properties = entry["properties"]
    properties["metadata"] = _metadata_property(model)
    for field_name, target in _identity_field_types(model).items():
        properties[field_name][FROM_IDENTITY_KEY] = target.__name__
    for field_name, (source, target, _) in _edge_field_types(model).items():
        properties[field_name][EDGE_KEY] = {"source": source.__name__, "target": target.__name__}


def _metadata_property(model: type[BaseModel]) -> dict:
    """The ``metadata`` property of a class's schema entry: its declared default."""
    return {
        "additionalProperties": True,
        "default": dict(model.model_fields["metadata"].default),
        "type": "object",
    }


if __name__ == "__main__":
    import asyncio
    from pprint import pprint

    import cognee
    from cognee.api.v1.search import SearchType
    from cognee.shared.logging_utils import ERROR, setup_logging

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
