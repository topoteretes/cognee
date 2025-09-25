import os
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts.render_prompt import render_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import (
    get_llm_config,
)


async def extract_event_entities(content: str, response_model: Type[BaseModel]):
    """
    Extracts event-related entities from the given content using an LLM with structured output.

    This function loads an event entity extraction prompt from the LLM configuration,
    renders it into a system prompt, and queries the LLM to produce structured entities
    that conform to the specified response model.

    Args:
        content (str): The input text from which to extract event entities.
        response_model (Type[BaseModel]): A Pydantic model defining the structure of the expected output.

    Returns:
        BaseModel: An instance of the response_model populated with extracted event entities.
    """
    llm_config = get_llm_config()

    prompt_path = llm_config.event_entity_prompt_path

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
