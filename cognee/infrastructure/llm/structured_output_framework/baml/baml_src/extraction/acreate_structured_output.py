import os
import asyncio
from typing import Type
from pydantic import BaseModel
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import SummarizedCode
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b
from cognee.infrastructure.llm.config import get_llm_config


logger = get_logger("extract_summary_baml")


async def acreate_structured_output(
    content: str, system_prompt: str, user_prompt: str, response_model: Type[BaseModel]
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

    # Use BAML's SummarizeContent function
    result = await b.AcreateStructuredOutput(
        content=content,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        baml_options={"client_registry": config.baml_registry},
    )

    return result


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(acreate_structured_output("TEST", SummarizedCode))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
