import asyncio
import re
import sys
import types
from pprint import pprint
from typing import Any, Union, cast, get_args, get_origin

from datamodel_code_generator import DataModelType, GenerateConfig, InputFileType, generate
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic._internal._core_utils import CoreSchemaOrField, is_core_schema
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticUndefined

import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import ERROR, setup_logging


def datapoint_model_to_basemodel(model: type[BaseModel]) -> type[BaseModel]:
    """
    Convert a DataPoint-derived model into a plain BaseModel-derived model at runtime.

    The converted model keeps only fields declared directly on each DataPoint subclass
    (excluding inherited DataPoint infrastructure fields).
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
        own_annotations = getattr(model_type, "__annotations__", {})

        # For DataPoint subclasses, keep only fields explicitly declared on the subclass.
        if issubclass(model_type, DataPoint):
            field_names = [name for name in own_annotations if name in model_fields]
        else:
            field_names = list(model_fields.keys())

        converted_fields = {}
        for field_name in field_names:
            field_info = model_fields[field_name]
            default_value = (
                Field(default_factory=field_info.default_factory)
                if field_info.default_factory is not None
                else field_info.default
            )
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

    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
