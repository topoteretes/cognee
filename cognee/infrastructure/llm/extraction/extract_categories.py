from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt


async def extract_categories(content: str, response_model: type[BaseModel]):
    system_prompt = read_query_prompt("classify_content.txt") or ""

    llm_output = await LLMGateway.acreate_structured_output(content, system_prompt, response_model)

    return llm_output
