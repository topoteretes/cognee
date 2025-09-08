import asyncio
from typing import Type
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import SummarizedCode
from cognee.infrastructure.llm.config import get_llm_config

from typing import List, Dict, Union, Optional, Literal
from enum import Enum
from baml_py import Image, Audio, Video, Pdf
from datetime import datetime

from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.type_builder import (
    TypeBuilder,
)
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client import b
from pydantic import BaseModel


logger = get_logger("extract_summary_baml")


def create_dynamic_baml_type(pydantic_model):
    tb = TypeBuilder()

    # if pydantic_model == str:
    #     b.ResponseModel.add_property("text", tb.string())
    #     return tb
    #
    # def map_type(field_type, field_info):
    #     # Handle Optional/Union types
    #     if getattr(field_type, "__origin__", None) == Union:
    #         # Extract types from Union
    #         types = field_type.__args__
    #         # Handle Optional (Union with NoneType)
    #         if type(None) in types:
    #             inner_type = next(t for t in types if t != type(None))
    #             return map_type(inner_type, field_info).optional()
    #         # Handle regular Union
    #         mapped_types = [map_type(t, field_info) for t in types]
    #         return tb.union(*mapped_types)
    #
    #     # Handle Lists
    #     if getattr(field_type, "__origin__", None) == list:
    #         inner_type = field_type.__args__[0]
    #         return map_type(inner_type, field_info).list()
    #
    #     # Handle Maps/Dictionaries
    #     if getattr(field_type, "__origin__", None) == dict:
    #         key_type, value_type = field_type.__args__
    #         # BAML only supports string or enum keys in maps
    #         if key_type not in [str, Enum]:
    #             raise ValueError("Map keys must be strings or enums in BAML")
    #         return tb.map(map_type(key_type, field_info), map_type(value_type, field_info))
    #
    #     # Handle Literal types
    #     if getattr(field_type, "__origin__", None) == Literal:
    #         literal_values = field_type.__args__
    #         return tb.union(*[tb.literal(val) for val in literal_values])
    #
    #     # Handle Enums
    #     if isinstance(field_type, type) and issubclass(field_type, Enum):
    #         enum_type = tb.add_enum(field_type.__name__)
    #         for member in field_type:
    #             enum_type.add_value(member.name)
    #         return enum_type.type()
    #
    #     # Handle primitive and special types
    #     type_mapping = {
    #         str: tb.string(),
    #         int: tb.int(),
    #         float: tb.float(),
    #         bool: tb.bool(),
    #         Image: tb.image(),
    #         Audio: tb.audio(),
    #         Video: tb.video(),
    #         Pdf: tb.pdf(),
    #         # datetime is not natively supported in BAML, map to string
    #         datetime: tb.string(),
    #     }
    #
    #     # Handle nested BaseModel classes
    #     if isinstance(field_type, type) and issubclass(field_type, BaseModel):
    #         nested_tb = create_dynamic_baml_type(field_type)
    #         # Get the last created class from the nested TypeBuilder
    #         return nested_tb.get_last_class().type()
    #
    #     if field_type in type_mapping:
    #         return type_mapping[field_type]
    #
    #     raise ValueError(f"Unsupported type: {field_type}")
    #
    # fields = pydantic_model.model_fields
    #
    # # Add fields
    # for field_name, field_info in fields.items():
    #     field_type = field_info.annotation
    #     baml_type = map_type(field_type, field_info)
    #
    #     # Add property with type
    #     prop = b.ResponseModel.add_property(field_name, baml_type)
    #
    #     # Add description if available
    #     if field_info.description:
    #         prop.description(field_info.description)

    return tb


async def acreate_structured_output(
    text_input: str, system_prompt: str, response_model: Type[BaseModel]
):
    """
    Extract summary using BAML framework.

    Args:
        content: The content to summarize
        response_model: The Pydantic model type for the response

    Returns:
        BaseModel: The summarized content in the specified format
    """
    config = get_llm_config()

    type_builder = create_dynamic_baml_type(response_model)

    result = await b.AcreateStructuredOutput(
        text_input=text_input,
        system_prompt=system_prompt,
        baml_options={"client_registry": config.baml_registry, "tb": type_builder},
    )

    return result


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(acreate_structured_output("TEST", SummarizedCode))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
