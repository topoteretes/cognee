import os
from typing import Any

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.graph_model_utils import datapoint_model_to_basemodel


async def extract_content_graph(
    content: str, response_model: type[BaseModel], custom_prompt: str | None = None, **kwargs: Any
) -> BaseModel:
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

    simplified_response_model = response_model
    if isinstance(response_model, type) and issubclass(response_model, DataPoint):
        simplified_response_model = datapoint_model_to_basemodel(
            response_model, strip_metadata=True
        )

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, simplified_response_model, **kwargs
    )

    if simplified_response_model is not response_model:
        return response_model.model_validate(content_graph.model_dump())
    return content_graph
