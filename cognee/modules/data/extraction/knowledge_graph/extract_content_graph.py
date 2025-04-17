from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from dotenv import load_dotenv
import os


async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()
    
# At the top of the file, after imports
load_dotenv()

async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()
    graph_prompt_path = os.getenv("GRAPH_PROMPT_PATH")
    # … rest of function …

    if graph_prompt_path is not None:
        path = graph_prompt_path
    else:
        path = "generate_graph_prompt.txt"

    system_prompt = render_prompt(path, {})
    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph
