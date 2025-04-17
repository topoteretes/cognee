from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.config import get_llm_config

async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()
    env_dict = get_llm_config().model_config

    if "GRAPH_PROMPT_PATH" in env_dict:
        path = env_dict["GRAPH_PROMPT_PATH"]
    else:
        path = "generate_graph_prompt.txt"

    system_prompt = render_prompt(path, {})
    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
