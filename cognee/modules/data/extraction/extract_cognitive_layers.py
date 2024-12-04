from typing import Type, Dict
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client

async def extract_cognitive_layers(content: str, category: Dict, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = render_prompt("generate_cog_layers.txt", category)

    return await llm_client.acreate_structured_output(content, system_prompt, response_model)
