import os
from pydantic import BaseModel
from typing import Type
from cognee.infrastructure.llm.LLMGateway import LLMGateway

from cognee.infrastructure.llm.config import (
    get_llm_config,
)


async def extract_event_graph(
    content: str, response_model: Type[BaseModel], system_prompt: str = None
):
    """Extract event graph from content using LLM."""

    llm_config = get_llm_config()

    prompt_path = llm_config.temporal_graph_prompt_path

    # Check if the prompt path is an absolute path or just a filename
    if os.path.isabs(prompt_path):
        # directory containing the file
        base_directory = os.path.dirname(prompt_path)
        # just the filename itself
        prompt_path = os.path.basename(prompt_path)
    else:
        base_directory = None

    system_prompt = LLMGateway.render_prompt(prompt_path, {}, base_directory=base_directory)

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
