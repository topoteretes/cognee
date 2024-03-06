""" Content to Propositions"""
from typing import Type
from pydantic import BaseModel
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from cognitive_architecture.shared.data_models import KnowledgeGraph

async def generate_graph(text_input:str,system_prompt_path:str, response_model: Type[BaseModel]):
    doc_path = "cognitive_architecture/infrastructure/llm/prompts/generate_graph_prompt.txt"
    llm_client = get_llm_client()

    return await llm_client.generate_graph(text_input,system_prompt_path, response_model)


if __name__ == "__main__":
    generate_graph("test", "test", response_model=KnowledgeGraph)

