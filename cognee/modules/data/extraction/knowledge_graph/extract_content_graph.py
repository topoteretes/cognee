from typing import Type, Optional
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt


async def extract_content_graph(
    content: str, response_model: Type[BaseModel], graph_prompt_path: Optional[str] = None
):
    llm_client = get_llm_client()

    prompt_path = graph_prompt_path or "generate_graph_prompt.txt"
    system_prompt = render_prompt(prompt_path, {})
    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
