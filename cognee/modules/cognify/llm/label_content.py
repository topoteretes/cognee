""" This module contains the code to classify content into categories using the LLM API. """
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client

async def label_content(text_input: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = read_query_prompt("label_content.txt")

    llm_output = await llm_client.acreate_structured_output(text_input, system_prompt, response_model)

    return llm_output.model_dump()
