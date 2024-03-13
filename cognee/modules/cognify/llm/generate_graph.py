""" This module is responsible for converting content to cognitive layers. """
import json
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.shared.data_models import KnowledgeGraph
from cognee.utils import async_render_template

async def generate_graph(text_input: str, filename: str, context, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    formatted_text_input = await async_render_template(filename, context)
    output = await llm_client.acreate_structured_output(text_input, formatted_text_input, response_model)

    context_key = json.dumps(context, sort_keys = True)

    # Returning a dictionary with context as the key and the awaited output as its value
    return {
        context_key: output
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_graph(text_input="bla", filename = "generate_graph_prompt.txt",context= {
        'layer': 'BLA'
    }, response_model=KnowledgeGraph))

