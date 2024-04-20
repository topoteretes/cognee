from typing import Type, List
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.get_llm_client import get_llm_client


async def extract_categories(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = read_query_prompt("classify_content.txt")

    llm_output = await llm_client.acreate_structured_output(content, system_prompt, response_model)

    return process_categories(llm_output.model_dump())

def process_categories(llm_output) -> List[dict]:
    # Extract the first subclass from the list (assuming there could be more)
    data_category = llm_output["label"]["subclass"][0]

    data_type = llm_output["label"]["type"].lower()

    return [{
        "data_type": data_type,
        # The data_category is the value of the Enum member (e.g., "News stories and blog posts")
        "category_name": data_category.value
    }]
