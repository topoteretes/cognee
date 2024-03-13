""" This module contains the code to classify content into categories using the LLM API. """
from typing import Type, List
from pydantic import BaseModel
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.utils import read_query_prompt

async def classify_into_categories(text_input: str, system_prompt_path: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = await read_query_prompt(system_prompt_path)

    llm_output = await llm_client.acreate_structured_output(text_input, system_prompt, response_model)

    return extract_categories(llm_output.dict())

def extract_categories(llm_output) -> List[dict]:
    # Extract the first subclass from the list (assuming there could be more)
    layer_enum = llm_output["label"]["subclass"][0]

    # The data type is derived from "type" and converted to lowercase
    data_type = llm_output["label"]["type"].lower()

    # The layer name is the name of the Enum member (e.g., "NEWS_STORIES")
    # layer_name = layer_enum.name.replace("_", " ").title()

    # The layer name is the value of the Enum member (e.g., "News stories and blog posts")
    layer_name = layer_enum.value

    return [{
        "data_type": data_type,
        "layer_name": layer_name  # llm layer classification
    }]

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(classify_into_categories("""Russia summons US ambassador in Moscow and says it will expel diplomats who meddle in its internal affairs
# The Russian foreign ministry said on Thursday it had summoned the US ambassador in Moscow and warned her against “attempts to interfere in the internal affairs of the Russian Federation”, reports Reuters.

# Ahead of a March presidential election, it said in a statement that such behaviour would be “firmly and resolutely suppressed, up to and including the expulsion as ‘persona non grata’ of US embassy staff involved in such actions”.""", "classify_content.txt", ContentPrediction))

