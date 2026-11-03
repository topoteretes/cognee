"""Custom graph models as JSON schema: a model to a schema, and a schema to a model.

This is the serialization boundary for custom graph models — what the HTTP API moves
across the wire. The LLM boundary (what an LLM is asked to answer for a model,
and how the answer becomes DataPoints again) lives in ``cognee.shared.llm_graph_model``.
"""

import re
import sys
import types
from typing import cast, get_args

from datamodel_code_generator import DataModelType, GenerateConfig, InputFileType, generate
from pydantic import BaseModel
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.FieldAnnotations import _FromIdentity
from cognee.shared.exceptions import ExternalSchemaReferenceError
from cognee.shared.llm_graph_model import (
    _contains_edge_annotation,
    _field_markers,
    _strip_annotated,
    datapoint_model_to_basemodel,
)


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

    # Return dynamically created Pydantic model class that can be used in cognee for graph creation and querying
    return graph_model


def fields_the_schema_cannot_carry(graph_model: type[BaseModel]) -> list[str]:
    """``Model.field`` names of the typed ``Edge[...]`` and ``FromIdentity`` fields in
    ``graph_model`` and every DataPoint type it nests.

    The schema only carries plain JSON types, so ``graph_schema_to_graph_model`` cannot
    bring these back: an edge row returns as a nested DataPoint with ``source`` and
    ``target`` strings and a reference as a plain ``str``. On the receiving side they
    become nodes and attributes, never edges and references.
    """
    found: list[str] = []
    seen: set[type] = set()

    def walk(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field_info in model.model_fields.items():
            if name in DataPoint.model_fields:
                continue
            if _contains_edge_annotation(field_info.annotation) or any(
                isinstance(marker, _FromIdentity) for marker in _field_markers(field_info)
            ):
                found.append(f"{model.__name__}.{name}")
            for nested in _datapoint_types_in(field_info.annotation):
                walk(nested)

    walk(graph_model)
    return found


def _datapoint_types_in(annotation) -> list[type[DataPoint]]:
    """Every DataPoint subclass named anywhere in ``annotation``."""
    core, _ = _strip_annotated(annotation)
    if isinstance(core, type):
        return [core] if issubclass(core, DataPoint) else []
    return [
        nested
        for arg in get_args(core)
        if arg is not type(None) and arg is not Ellipsis
        for nested in _datapoint_types_in(arg)
    ]


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
