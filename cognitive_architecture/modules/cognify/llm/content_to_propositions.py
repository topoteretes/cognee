""" Content to Propositions"""
from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client


async def generate_graph(memory_name: str, payload: str):
    doc_path = "cognitive_architecture/infrastructure/llm/prompts/generate_graph_prompt.txt"
    llm_client = get_llm_client()
    return await llm_client.generate_graph(memory_name,  doc_path=doc_path,payload= payload)

