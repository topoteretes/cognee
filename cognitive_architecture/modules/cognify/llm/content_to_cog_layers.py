""" This module contains the code to classify content into categories using the LLM API. """
from typing import Type
from pydantic import BaseModel
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.shared.data_models import CognitiveLayer
from cognitive_architecture.utils import async_render_template

async def content_to_cog_layers(filename: str,context, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    formatted_text_input = await async_render_template(filename, context)


    # data_points = list()
    # for point in map(create_data_point, payload):
    #     data_points.append(await point)

    return await llm_client.acreate_structured_output(formatted_text_input,formatted_text_input, response_model)

if __name__ == "__main__":
    import asyncio
    asyncio.run(content_to_cog_layers("generate_cog_layers.txt", {
        'data_type': 'text',
        'context_name': 'Scientific Research',
        'layer_name': 'Content Layer'
    }, response_model=CognitiveLayer))




