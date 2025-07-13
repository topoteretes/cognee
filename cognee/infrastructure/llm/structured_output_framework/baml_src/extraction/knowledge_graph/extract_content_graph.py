import os
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.structured_output_framework.baml_src.config import get_llm_config

config = get_llm_config()
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b
from cognee.shared.logging_utils import get_logger, setup_logging
from baml_py import ClientRegistry


async def extract_content_graph(
    content: str, response_model: Type[BaseModel], mode: str = "simple"
):
    config = get_llm_config()
    setup_logging()

    get_logger(level="INFO")

    baml_registry = ClientRegistry()

    baml_registry.add_llm_client(
        name="extract_content_client",
        provider=config.llm_provider,
        options={
            "model": config.llm_model,
            "temperature": config.llm_temperature,
            "api_key": config.llm_api_key,
        },
    )
    baml_registry.set_primary("extract_content_client")

    # if response_model:
    #     # tb = TypeBuilder()
    #     # country = tb.union \
    #     #     ([tb.literal_string("USA"), tb.literal_string("UK"), tb.literal_string("Germany"), tb.literal_string("other")])
    #     # tb.Node.add_property("country", country)
    #
    #     graph = await b.ExtractDynamicContentGraph(
    #         content, mode=mode, baml_options={"client_registry": baml_registry}
    #     )
    #
    #     return graph

    # else:
    graph = await b.ExtractContentGraph(
        content, mode=mode, baml_options={"client_registry": baml_registry}
    )

    return graph
