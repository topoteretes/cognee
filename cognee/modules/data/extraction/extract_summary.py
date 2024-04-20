from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client

async def extract_summary(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = read_query_prompt("summarize_content.txt")

    llm_output = await llm_client.acreate_structured_output(content, system_prompt, response_model)

    return llm_output.model_dump()
