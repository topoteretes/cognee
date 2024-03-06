from pydantic import BaseModel
from typing import Type
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.shared.data_models import ContentPrediction

async def content_to_cog_layers(text_input: str,system_prompt_path:str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    # data_points = list()
    # for point in map(create_data_point, payload):
    #     data_points.append(await point)

    return await llm_client.acreate_structured_output(text_input,system_prompt_path, response_model)


if __name__ == "__main__":

    content_to_cog_layers("test", "test", ContentPrediction)


