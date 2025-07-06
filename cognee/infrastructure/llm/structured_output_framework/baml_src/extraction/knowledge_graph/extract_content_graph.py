import os
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.type_builder import (
    TypeBuilder,
)
from cognee.infrastructure.llm.structured_output_framework.baml_src.config import get_llm_config

from cognee.shared.logging_utils import get_logger, setup_logging


async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    # tb = TypeBuilder()
    config = get_llm_config()
    setup_logging()
    get_logger(level="INFO")
    # country = tb.union \
    #     ([tb.literal_string("USA"), tb.literal_string("UK"), tb.literal_string("Germany"), tb.literal_string("other")])
    # tb.Node.add_property("country", country)

    graph = await b.ExtractContentGraph(
        content, mode="simple", baml_options={"tb": config.baml_registry}
    )

    return graph
