import os
from pydantic import BaseModel
from typing import Type

from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import (
    get_llm_config,
)


async def extract_event_graph(content: str, response_model: Type[BaseModel]):
    """
    Extracts an event graph from the given content using an LLM with a structured output format.

    This function loads a temporal graph extraction prompt from the LLM configuration,
    renders it as a system prompt, and queries the LLM to produce a structured event
    graph matching the specified response model.

    Args:
        content (str): The input text from which to extract the event graph.
        response_model (Type[BaseModel]): A Pydantic model defining the structure of the expected output.

    Returns:
        BaseModel: An instance of the response_model populated with the extracted event graph.
    """

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

    system_prompt = render_prompt(prompt_path, {}, base_directory=base_directory)

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
