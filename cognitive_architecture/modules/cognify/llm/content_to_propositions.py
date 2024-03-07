""" This module is responsible for converting content to cognitive layers. """
from typing import Type
from pydantic import BaseModel
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.shared.data_models import KnowledgeGraph
from cognitive_architecture.utils import async_render_template

async def generate_graph(filename: str,context, response_model: Type[BaseModel]):

    llm_client = get_llm_client()

    formatted_text_input = await async_render_template(filename, context)
    return await llm_client.acreate_structured_output(formatted_text_input,formatted_text_input, response_model)


if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_graph("generate_graph_prompt.txt", {
        'layer': 'text'
    }, response_model=KnowledgeGraph))

