""" This module contains the code to classify content into categories using the LLM API. """
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client

async def content_to_cog_layers(context, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    formatted_text_input = render_prompt("generate_cog_layers.txt", context)

    return await llm_client.acreate_structured_output(formatted_text_input, formatted_text_input, response_model)
