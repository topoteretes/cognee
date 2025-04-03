from typing import Type, Optional
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.config import get_llm_config


async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()
    llm_config = get_llm_config()

    prompt_path = llm_config.graph_prompt_path
    system_prompt = render_prompt(prompt_path, {})
    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
