from typing import Type
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway


async def extract_categories(content: str, response_model: Type[BaseModel]):
    system_prompt = LLMGateway.read_query_prompt("classify_content.txt")

    llm_output = await LLMGateway.acreate_structured_output(content, system_prompt, response_model)

    return llm_output
