"""Custom graph models as JSON schema: a model to a schema, and a schema to a model.

This is the serialization boundary for custom graph models — what the HTTP API moves
across the wire. The LLM boundary (what an LLM is asked to answer for a model,
and how the answer becomes DataPoints again) lives in ``cognee.shared.llm_graph_model``.
"""

import copy
import re
import sys
import types
from typing import Annotated, Any, Union, cast, get_args, get_origin

from datamodel_code_generator import (
    DataModelType,
    GenerateConfig,
    InputFileType,
    LiteralType,
    generate,
)
from pydantic import BaseModel
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema

from cognee.infrastructure.engine import DataPoint, Edge, FromIdentity
from cognee.shared.exceptions import ExternalSchemaReferenceError
from cognee.shared.llm_graph_model import (
    _edge_field_types,
    _identity_field_types,
    _is_union,
    _strip_annotated,
    datapoint_model_to_basemodel,
)

# Graph semantics the LLM-facing schema cannot express, carried as JSON-schema keywords.
# On a model's schema: the DataPoint ``metadata`` default (index and identity fields).
METADATA_KEYWORD = "x-cognee-metadata"
# On a property: the DataPoint class a FromIdentity reference names.
REFERENCE_KEYWORD = "x-cognee-reference"
# On a property: the endpoint classes of a list[Edge[...]] field.
EDGE_KEYWORD = "x-cognee-edge"


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
        # A generated Enum class would reach the graph store as a value no JSON
        # encoder accepts; a Literal keeps the plain string the LLM answered.
        enum_field_as_literal=LiteralType.All,
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

    _restore_graph_semantics(pydantic_json_schema, namespace)

    # Return dynamically created Pydantic model class that can be used in cognee for graph creation and querying
    return graph_model


def _restore_graph_semantics(schema: dict, namespace: dict) -> None:
    """Put back what ``graph_model_to_graph_schema`` recorded as ``x-cognee-*`` keywords.

    A schema without them (hand-written, or from the graph model editor) is left as
    generated.
    """
    model_schemas = [(schema["title"], schema), *schema.get("$defs", {}).items()]
    patched = []
    for name, model_schema in model_schemas:
        model = namespace.get(name)
        if not (isinstance(model, type) and issubclass(model, DataPoint)):
            continue
        fields = {}
        if METADATA_KEYWORD in model_schema:
            fields["metadata"] = _with(
                model.model_fields["metadata"], default=model_schema[METADATA_KEYWORD]
            )
        for field_name, field_schema in model_schema.get("properties", {}).items():
            if REFERENCE_KEYWORD in field_schema:
                target = namespace[field_schema[REFERENCE_KEYWORD]]
                field_info = model.model_fields[field_name]
                fields[field_name] = _with(
                    field_info,
                    annotation=_reference_annotation(
                        field_info.annotation, Annotated[target, FromIdentity()]
                    ),
                )
            if EDGE_KEYWORD in field_schema:
                endpoints = field_schema[EDGE_KEYWORD]
                field_info = model.model_fields[field_name]
                fields[field_name] = _with(
                    field_info,
                    annotation=_edge_annotation(
                        field_info.annotation,
                        namespace[endpoints["source"]],
                        namespace[endpoints["target"]],
                    ),
                )
        if fields:
            # Copies, never in place: an inherited FieldInfo (metadata) is DataPoint's own.
            model.__pydantic_fields__.update(fields)
            patched.append(model)

    for model in patched:
        model.model_rebuild(force=True, _types_namespace=namespace)


def _with(field_info: Any, **changes: Any) -> Any:
    changed = copy.copy(field_info)
    for attribute, value in changes.items():
        setattr(changed, attribute, value)
    return changed


def _reference_annotation(annotation: Any, reference: Any) -> Any:
    """``annotation`` with the identity string it holds replaced by ``reference``."""
    core, _ = _strip_annotated(annotation)
    if core is str:
        return reference
    if get_origin(core) is list:
        return list[_reference_annotation(get_args(core)[0], reference)]
    if _is_union(get_origin(core)):
        return Union[  # noqa: UP007 - built from a runtime tuple; no | spelling
            tuple(
                arg if arg is type(None) else _reference_annotation(arg, reference)
                for arg in get_args(core)
            )
        ]
    raise ValueError(f"{REFERENCE_KEYWORD} marks {annotation!r}, which holds no identity string")


def _edge_annotation(annotation: Any, source_type: type, target_type: type) -> Any:
    """``list[Edge[source, target, ...]]`` in place of the edge rows ``annotation`` holds."""
    core, _ = _strip_annotated(annotation)
    if _is_union(get_origin(core)):
        return Union[  # noqa: UP007 - built from a runtime tuple; no | spelling
            tuple(
                arg if arg is type(None) else _edge_annotation(arg, source_type, target_type)
                for arg in get_args(core)
            )
        ]
    if get_origin(core) is not list:
        raise ValueError(f"{EDGE_KEYWORD} marks {annotation!r}, which is not a list of edge rows")
    row_model, _ = _strip_annotated(get_args(core)[0])
    relationship = row_model.model_fields.get("relationship_type")
    if relationship is None:
        return list[Edge[source_type, target_type]]
    return list[Edge[source_type, target_type, relationship.annotation]]


class _GenerateJsonSchemaWithoutDefaultTitles(GenerateJsonSchema):
    def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
        return_value = super().field_title_should_be_set(schema)
        if return_value and is_core_schema(schema):
            return False
        return return_value


def _llm_json_schema(model: type[BaseModel]) -> dict:
    return datapoint_model_to_basemodel(model).model_json_schema(
        schema_generator=_GenerateJsonSchemaWithoutDefaultTitles
    )


def graph_model_to_graph_schema(graph_model: type[BaseModel]) -> dict:
    """The JSON schema of the model the LLM answers, plus the graph semantics it drops.

    The LLM-facing schema narrows ``FromIdentity`` references to strings, replaces typed
    ``Edge`` fields by flat rows and leaves out every DataPoint's ``metadata``. Each of
    those is recorded as an ``x-cognee-*`` keyword next to what it describes, so
    ``graph_schema_to_graph_model`` rebuilds the same model on the other side of the wire.
    """
    schema = _llm_json_schema(graph_model)
    if isinstance(graph_model, type) and issubclass(graph_model, DataPoint):
        _annotate_graph_semantics(schema, graph_model)
    return schema


def _annotate_graph_semantics(schema: dict, graph_model: type[DataPoint]) -> None:
    definitions = schema.setdefault("$defs", {})
    pending = [graph_model]
    seen: set[type[DataPoint]] = set()
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)

        if model is graph_model:
            model_schema = schema
        elif model.__name__ in definitions:
            model_schema = definitions[model.__name__]
        else:
            # Only reachable as a reference or an edge endpoint, so the LLM schema
            # names it nowhere; the receiving side still needs the node class.
            model_schema = _llm_json_schema(model)
            definitions.update(model_schema.pop("$defs", {}))
            definitions[model.__name__] = model_schema

        model_schema[METADATA_KEYWORD] = model.model_fields["metadata"].default
        properties = model_schema["properties"]
        references = _identity_field_types(model)
        edges = _edge_field_types(model)
        for name, target_type in references.items():
            properties[name][REFERENCE_KEYWORD] = target_type.__name__
            pending.append(target_type)
        for name, (source_type, target_type, _) in edges.items():
            properties[name][EDGE_KEYWORD] = {
                "source": source_type.__name__,
                "target": target_type.__name__,
            }
            pending.extend((source_type, target_type))
        for name, field_info in model.model_fields.items():
            if name in DataPoint.model_fields or name in references or name in edges:
                continue
            pending.extend(_datapoint_classes_in(field_info.annotation))


def _datapoint_classes_in(annotation: Any) -> list[type[DataPoint]]:
    core, _ = _strip_annotated(annotation)
    if isinstance(core, type) and issubclass(core, DataPoint):
        return [core]
    return [found for arg in get_args(core) for found in _datapoint_classes_in(arg)]


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
