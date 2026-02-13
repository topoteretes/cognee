import asyncio
from pprint import pprint

import cognee
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.api.v1.search import SearchType

import typing
from typing import Any

# async def json_schema_to_pydantic_model(json_schema: dict) -> Any:


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    Python is the best programming language in Data science.
    """

    print("Adding text to cognee:")
    print(text.strip())
    # Add the text, and make it available for cognify
    await cognee.add(text)
    print("Text added successfully.\n")

    print("Running cognify to create knowledge graph...\n")
    print("Cognify process steps:")
    print("1. Classifying the document: Determining the type and category of the input text.")
    print(
        "2. Extracting text chunks: Breaking down the text into sentences or phrases for analysis."
    )
    print(
        "3. Generating knowledge graph: Extracting entities and relationships to form a knowledge graph."
    )
    print("4. Summarizing text: Creating concise summaries of the content for quick insights.")
    print("5. Adding data points: Storing the extracted chunks for processing.\n")

    from pydantic import BaseModel
    from pydantic._internal._core_utils import is_core_schema, CoreSchemaOrField
    from pydantic.json_schema import GenerateJsonSchema

    class GenerateJsonSchemaWithoutDefaultTitles(GenerateJsonSchema):
        def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
            return_value = super().field_title_should_be_set(schema)
            if return_value and is_core_schema(schema):
                return False
            return return_value

    import os
    import typing
    import asyncio
    from uuid import UUID
    from pydantic import Field
    from typing import List, Optional
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
    from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION
    from datamodel_code_generator import InputFileType, generate, GenerateConfig, DataModelType

    from cognee.api.DTO import InDTO
    from cognee.modules.pipelines.methods import get_pipeline_run
    from cognee.modules.users.models import User
    from cognee.modules.users.methods import get_authenticated_user
    from cognee.modules.users.get_user_db import get_user_db_context
    from cognee.modules.graph.methods import get_formatted_graph_data
    from cognee.modules.users.get_user_manager import get_user_manager_context
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.authentication.default.default_jwt_strategy import DefaultJWTStrategy
    from cognee.shared.data_models import KnowledgeGraph
    from cognee.modules.pipelines.models.PipelineRunInfo import (
        PipelineRunCompleted,
        PipelineRunInfo,
        PipelineRunErrored,
    )
    from cognee.modules.pipelines.queues.pipeline_run_info_queues import (
        get_from_queue,
        initialize_queue,
        remove_queue,
    )
    from cognee.shared.logging_utils import get_logger
    from cognee.shared.utils import send_telemetry
    from cognee.shared.usage_logger import log_usage
    from cognee import __version__ as cognee_version
    from cognee.infrastructure.engine import DataPoint

    # Define a custom graph model for programming languages.
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

    graph_model = ProgrammingLanguage.model_json_schema(
        schema_generator=GenerateJsonSchemaWithoutDefaultTitles
    )
    # If a custom graph model is provided, convert it from dict to a Pydantic model class
    config = GenerateConfig(
        input_file_type=InputFileType.JsonSchema,
        input_filename="example.json",
        output_model_type=DataModelType.PydanticV2BaseModel,
        additional_imports=["cognee.infrastructure.engine.DataPoint", "typing.Any", "typing"],
        base_class="cognee.infrastructure.engine.DataPoint",
        type_overrides={"DataPoint": "cognee.infrastructure.engine.DataPoint"},
    )
    # Override title to ensure a valid and secure Python class name for the generated model
    graph_model["title"] = "DynamicGraphModel"
    result = generate(graph_model, config=config)
    import re

    result = re.sub(
        r"class DataPointModel\(DataPoint\):.*?(?=\nclass|\Z)", "", result, flags=re.DOTALL
    )
    # Replace all remaining references
    result = result.replace("DataPointModel", "DataPoint")
    from typing import Any

    import sys
    import types
    module_name = "cognee.shared._generated_graph_models"
    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod

    exec(result, mod.__dict__)
    namespace = mod.__dict__

    # namespace = {"Any": Any, "typing": typing}
    # exec(result, namespace)
    graph_model = namespace[graph_model["title"]]
    # Rebuild the base class first
    namespace["DataPoint"].model_rebuild()  # Resolves DataPoint's self-reference
    graph_model.model_rebuild(_types_namespace=namespace)

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify(graph_model=graph_model)
    print("Cognify process complete.\n")

    query_text = "Tell me about NLP"
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


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
