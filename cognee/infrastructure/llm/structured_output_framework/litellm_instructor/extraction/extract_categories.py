from typing import Type
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMAdapter import LLMAdapter


async def extract_categories(content: str, response_model: Type[BaseModel]):
    system_prompt = LLMAdapter.read_query_prompt("classify_content.txt")

    llm_output = await LLMAdapter.acreate_structured_output(content, system_prompt, response_model)

    return llm_output
