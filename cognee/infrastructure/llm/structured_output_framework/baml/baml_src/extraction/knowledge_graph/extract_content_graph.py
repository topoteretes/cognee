from typing import Type, Optional
from pydantic import BaseModel
from cognee.infrastructure.llm.config import get_llm_config
from cognee.shared.logging_utils import get_logger, setup_logging
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b


async def extract_content_graph(
    content: str,
    response_model: Type[BaseModel],
    mode: str = "simple",
    custom_prompt: Optional[str] = None,
):
    config = get_llm_config()
    setup_logging()

    get_logger(level="INFO")

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
    if custom_prompt:
        graph = await b.ExtractContentGraphGeneric(
            content,
            mode="custom",
            custom_prompt_content=custom_prompt,
            baml_options={"client_registry": config.baml_registry},
        )
    else:
        graph = await b.ExtractContentGraphGeneric(
            content, mode=mode, baml_options={"client_registry": config.baml_registry}
        )

    return graph
