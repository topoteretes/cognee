import os
from typing import Type, Optional
from pydantic import BaseModel

from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import (
    get_llm_config,
)


async def extract_content_graph(
    content: str, response_model: Type[BaseModel], custom_prompt: Optional[str] = None
):
    if custom_prompt:
        system_prompt = custom_prompt
    else:
        llm_config = get_llm_config()
        prompt_path = llm_config.graph_prompt_path

        # Check if the prompt path is an absolute path or just a filename
        if os.path.isabs(prompt_path):
            # directory containing the file
            base_directory = os.path.dirname(prompt_path)
            # just the filename itself
            prompt_path = os.path.basename(prompt_path)
        else:
            base_directory = None

        system_prompt = render_prompt(prompt_path, {}, base_directory=base_directory)

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
