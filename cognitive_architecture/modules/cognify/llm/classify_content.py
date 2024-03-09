""" This module contains the code to classify content into categories using the LLM API. """
from pydantic import BaseModel
from typing import Type
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.shared.data_models import DefaultContentPrediction
from cognitive_architecture.utils import read_query_prompt

async def classify_into_categories(text_input: str, system_prompt_path:str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = await read_query_prompt(system_prompt_path)

    # data_points = list()
    # for point in map(create_data_point, payload):
    #     data_points.append(await point)

    return await llm_client.acreate_structured_output(text_input,system_prompt, response_model)




# Your async function definitions and other code here...

if __name__ == "__main__":
    import asyncio
    asyncio.run(classify_into_categories("""Russia summons US ambassador in Moscow and says it will expel diplomats who meddle in its internal affairs
The Russian foreign ministry said on Thursday it had summoned the US ambassador in Moscow and warned her against “attempts to interfere in the internal affairs of the Russian Federation”, reports Reuters.

Ahead of a March presidential election, it said in a statement that such behaviour would be “firmly and resolutely suppressed, up to and including the expulsion as ‘persona non grata’ of US embassy staff involved in such actions”.""", "classify_content.txt", ContentPrediction))

